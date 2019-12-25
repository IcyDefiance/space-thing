pub use ash::vk::BufferUsageFlags;

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
		unsafe {
			let len = data.len();
			let size = size_of::<T>() as u64 * len as u64;

			let create_info = AllocationCreateInfo { usage: MemoryUsage::CpuOnly, ..Default::default() };
			let (cpubuf, cpualloc, _) = gfx
				.allocator
				.create_buffer(
					&ash::vk::BufferCreateInfo::builder().size(size).usage(vk::BufferUsageFlags::TRANSFER_SRC).build(),
					&create_info,
				)
				.unwrap();

			let bufdata = gfx.allocator.map_memory(&cpualloc).unwrap();
			let bufdata = slice::from_raw_parts_mut(bufdata as *mut T, (size / size_of::<T>() as u64) as _);
			bufdata.clone_from_slice(data);
			gfx.allocator.unmap_memory(&cpualloc).unwrap();

			let create_info = AllocationCreateInfo { usage: MemoryUsage::CpuOnly, ..Default::default() };
			let (buf, allocation, _) = gfx
				.allocator
				.create_buffer(
					&ash::vk::BufferCreateInfo::builder()
						.size(size)
						.usage(usage | vk::BufferUsageFlags::TRANSFER_DST)
						.build(),
					&create_info,
				)
				.unwrap();

			let fence = gfx.device.create_fence(&vk::FenceCreateInfo::builder(), None).unwrap();

			let ci = vk::CommandBufferAllocateInfo::builder()
				.command_pool(gfx.cmdpool_transient)
				.level(vk::CommandBufferLevel::PRIMARY)
				.command_buffer_count(1);
			let cmds = gfx.device.allocate_command_buffers(&ci).unwrap();
			let cmd = cmds[0];
			gfx.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder()).unwrap();
			gfx.device.cmd_copy_buffer(cmd, cpubuf, buf, &[vk::BufferCopy::builder().size(size).build()]);
			gfx.device.end_command_buffer(cmd).unwrap();

			let submits = [vk::SubmitInfo::builder().command_buffers(&cmds).build()];
			gfx.device.queue_submit(gfx.queue, &submits, fence).unwrap();

			gfx.device.wait_for_fences(&[fence], false, !0).unwrap();

			gfx.device.destroy_fence(fence, None);
			gfx.device.free_command_buffers(gfx.cmdpool_transient, &cmds);
			gfx.device.destroy_buffer(cpubuf, None);
			gfx.allocator.free_memory(&cpualloc).unwrap();

			Self { gfx, buf, allocation, len, _phantom: PhantomData }
		}
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
