pub use ash::vk::BufferUsageFlags;
use ash::Device;
use vk_mem::Allocator;

use crate::gfx::Gfx;
use ash::{version::DeviceV1_0, vk};
use std::{marker::PhantomData, mem::size_of, slice, sync::Arc, u64};
use vk_mem::{Allocation, AllocationCreateInfo, MemoryUsage};

pub struct ImmutableBuffer<T: ?Sized> {
	pub(super) gfx: Arc<Gfx>,
	pub(super) buf: vk::Buffer,
	allocation: Allocation,
	pub(super) len: usize,
	_phantom: PhantomData<Box<T>>,
}
impl<T: Clone> ImmutableBuffer<[T]> {
	pub fn from_slice(gfx: Arc<Gfx>, data: &[T], usage: BufferUsageFlags) -> Self {
		let (buf, allocation) =
			create_device_local_buffer(&gfx.device, gfx.queue, &gfx.allocator, gfx.cmdpool_transient, data, usage);

		Self { gfx, buf, allocation, len: data.len(), _phantom: PhantomData }
	}
}
impl<T: ?Sized> Drop for ImmutableBuffer<T> {
	fn drop(&mut self) {
		unsafe {
			self.gfx.device.destroy_buffer(self.buf, None);
			self.gfx.allocator.free_memory(&self.allocation).unwrap();
		}
	}
}

pub(super) fn create_device_local_buffer<T: Clone>(
	device: &Device,
	queue: vk::Queue,
	allocator: &Allocator,
	cmdpool: vk::CommandPool,
	data: &[T],
	usage: BufferUsageFlags,
) -> (vk::Buffer, Allocation) {
	unsafe {
		let size = size_of::<T>() as u64 * data.len() as u64;

		let ci = ash::vk::BufferCreateInfo::builder().size(size).usage(vk::BufferUsageFlags::TRANSFER_SRC).build();
		let aci = AllocationCreateInfo { usage: MemoryUsage::CpuOnly, ..Default::default() };
		let (cpubuf, cpualloc, _) = allocator.create_buffer(&ci, &aci).unwrap();

		let bufdata = allocator.map_memory(&cpualloc).unwrap();
		let bufdata = slice::from_raw_parts_mut(bufdata as *mut T, (size / size_of::<T>() as u64) as _);
		bufdata.clone_from_slice(data);
		allocator.unmap_memory(&cpualloc).unwrap();

		let ci =
			ash::vk::BufferCreateInfo::builder().size(size).usage(usage | vk::BufferUsageFlags::TRANSFER_DST).build();
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
