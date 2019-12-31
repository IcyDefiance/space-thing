use crate::{
	command::CommandPool,
	device::{Device, Queue, SubmitFuture},
};
use ash::{version::DeviceV1_0, vk};
use std::{marker::PhantomData, mem::size_of, slice, sync::Arc};
use typenum::B1;
use vk_mem::Allocation;

pub struct Buffer<T: ?Sized> {
	device: Arc<Device>,
	pub vk: vk::Buffer,
	alloc: Allocation,
	size: u64,
	phantom: PhantomData<T>,
}
impl<T: ?Sized> Buffer<T> {
	pub fn size(&self) -> u64 {
		self.size
	}

	pub(crate) fn from_vk(device: Arc<Device>, vk: vk::Buffer, alloc: Allocation, size: u64) -> Arc<Self> {
		Arc::new(Self { device, vk, alloc, size, phantom: PhantomData })
	}
}
impl<T: ?Sized> Drop for Buffer<T> {
	fn drop(&mut self) {
		unsafe { self.device.vk.destroy_buffer(self.vk, None) };
		self.device.allocator.free_memory(&self.alloc).unwrap();
	}
}
impl<T: ?Sized> BufferAbstract for Buffer<T> {}

pub struct BufferInit<T: ?Sized, CPU> {
	buf: Arc<Buffer<T>>,
	phantom: PhantomData<CPU>,
}
impl<T: ?Sized, CPU> BufferInit<T, CPU> {
	pub fn from_vk(device: Arc<Device>, vk: vk::Buffer, alloc: Allocation, size: u64) -> Self {
		Self { buf: Buffer::from_vk(device, vk, alloc, size), phantom: PhantomData }
	}
}
impl<T: 'static, CPU> BufferInit<[T], CPU> {
	pub fn copy_from_buffer(
		self,
		queue: &Arc<Queue>,
		pool: &Arc<CommandPool>,
		buffer: Arc<Buffer<[T]>>,
	) -> (Arc<Buffer<[T]>>, SubmitFuture) {
		let cmd = pool.allocate_command_buffers(false, 1).next().unwrap();
		cmd.record(|cmd| cmd.copy_buffer(buffer, self.buf.clone()));

		let future = queue.submit(cmd);
		(self.buf, future)
	}
}
impl<T: Copy + 'static> BufferInit<[T], B1> {
	pub fn copy_from_slice(self, data: &[T]) -> Arc<Buffer<[T]>> {
		let buf = self.buf;
		let allocator = &buf.device.allocator;
		let alloc = &buf.alloc;

		let bufdata = allocator.map_memory(&alloc).unwrap();
		let bufdata = unsafe { slice::from_raw_parts_mut(bufdata as *mut T, (buf.size / size_of::<T>() as u64) as _) };
		bufdata.copy_from_slice(data);
		allocator.unmap_memory(&alloc).unwrap();

		buf
	}
}

pub(crate) trait BufferAbstract {}
