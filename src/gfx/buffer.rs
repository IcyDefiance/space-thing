pub use ash::vk::BufferUsageFlags;

use ash::{version::DeviceV1_0, vk, Device};
use std::{mem::size_of, slice, u64};
use vk_mem::{Allocation, AllocationCreateInfo, Allocator, MemoryUsage};

pub(super) fn create_cpu_buffer<T>(allocator: &Allocator, len: usize) -> (vk::Buffer, Allocation, &'static mut [T]) {
	unsafe {
		let size = size_of::<T>() as u64 * len as u64;

		let ci = ash::vk::BufferCreateInfo::builder().size(size).usage(vk::BufferUsageFlags::TRANSFER_SRC);
		let aci = AllocationCreateInfo { usage: MemoryUsage::CpuOnly, ..Default::default() };
		let (buf, alloc, _) = allocator.create_buffer(&ci, &aci).unwrap();

		let bufdata = allocator.map_memory(&alloc).unwrap();
		let bufdata = slice::from_raw_parts_mut(bufdata as _, len as _);

		(buf, alloc, bufdata)
	}
}

pub(super) fn create_device_local_buffer<T: Copy + 'static>(
	device: &Device,
	queue: vk::Queue,
	allocator: &Allocator,
	cmdpool: vk::CommandPool,
	data: &[T],
	usage: BufferUsageFlags,
) -> (vk::Buffer, Allocation) {
	unsafe {
		let size = size_of::<T>() as u64 * data.len() as u64;

		let (cpubuf, cpualloc, cpumap) = create_cpu_buffer::<T>(allocator, data.len());
		cpumap.copy_from_slice(data);
		allocator.unmap_memory(&cpualloc).unwrap();

		let ci = ash::vk::BufferCreateInfo::builder().size(size).usage(usage | vk::BufferUsageFlags::TRANSFER_DST);
		let aci = AllocationCreateInfo { usage: MemoryUsage::CpuOnly, ..Default::default() };
		let (buf, allocation, _) = allocator.create_buffer(&ci, &aci).unwrap();

		let fence = device.create_fence(&vk::FenceCreateInfo::builder(), None).unwrap();

		let ci = vk::CommandBufferAllocateInfo::builder()
			.command_pool(cmdpool)
			.level(vk::CommandBufferLevel::PRIMARY)
			.command_buffer_count(1);
		let cmds = device.allocate_command_buffers(&ci).unwrap();
		let cmd = cmds[0];
		device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder()).unwrap();
		device.cmd_copy_buffer(cmd, cpubuf, buf, &[vk::BufferCopy::builder().size(size).build()]);
		device.end_command_buffer(cmd).unwrap();

		let submits = [vk::SubmitInfo::builder().command_buffers(&cmds).build()];
		device.queue_submit(queue, &submits, fence).unwrap();

		device.wait_for_fences(&[fence], false, !0).unwrap();

		device.destroy_fence(fence, None);
		device.free_command_buffers(cmdpool, &cmds);
		device.destroy_buffer(cpubuf, None);
		allocator.free_memory(&cpualloc).unwrap();

		(buf, allocation)
	}
}
