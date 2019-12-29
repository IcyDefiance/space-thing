use crate::{
	buffer::{Buffer, BufferAbstract},
	device::Device,
};
use ash::{version::DeviceV1_0, vk};
use std::sync::{Arc, Mutex, RwLock};

pub struct CommandPool {
	device: Arc<Device>,
	pub(crate) queue_family: u32,
	pub vk: Mutex<vk::CommandPool>,
	free: Mutex<Vec<vk::CommandBuffer>>,
}
impl CommandPool {
	pub fn allocate_command_buffers<'a>(
		self: &'a Arc<Self>,
		secondary: bool,
		count: u32,
	) -> impl Iterator<Item = Arc<CommandBuffer>> + 'a {
		let level = if secondary { vk::CommandBufferLevel::SECONDARY } else { vk::CommandBufferLevel::PRIMARY };

		let vk = self.vk.lock().unwrap();
		let ci = vk::CommandBufferAllocateInfo::builder().command_pool(*vk).level(level).command_buffer_count(count);
		unsafe { self.device.vk.allocate_command_buffers(&ci) }
			.unwrap()
			.into_iter()
			.map(move |vk| unsafe { CommandBuffer::from_vk(self.clone(), vk) })
	}

	pub fn trim(&self) {
		let mut free = self.free.lock().unwrap();
		unsafe { self.device.vk.free_command_buffers(*self.vk.lock().unwrap(), &*free) };
		free.clear();
	}

	pub(crate) unsafe fn from_vk(device: Arc<Device>, queue_family: u32, vk: vk::CommandPool) -> Arc<Self> {
		Arc::new(Self { device, queue_family, vk: Mutex::new(vk), free: Mutex::default() })
	}
}
impl Drop for CommandPool {
	fn drop(&mut self) {
		unsafe { self.device.vk.destroy_command_pool(*self.vk.get_mut().unwrap(), None) };
	}
}

pub struct CommandBuffer {
	pub(crate) pool: Arc<CommandPool>,
	pub(crate) inner: RwLock<CommandBufferInner>,
}
impl CommandBuffer {
	pub fn record(&self, cb: impl FnOnce(&mut CommandBufferRecording)) {
		let _pool = self.pool.vk.lock().unwrap();
		let mut inner = self.inner.write().unwrap();

		unsafe {
			self.pool.device.vk.begin_command_buffer(inner.vk, &vk::CommandBufferBeginInfo::builder()).unwrap();
			{
				let mut rec = CommandBufferRecording { device: &self.pool.device, inner: &mut inner };
				cb(&mut rec);
			}
			self.pool.device.vk.end_command_buffer(inner.vk).unwrap();
		}
	}

	unsafe fn from_vk(pool: Arc<CommandPool>, vk: vk::CommandBuffer) -> Arc<Self> {
		Arc::new(Self { pool, inner: RwLock::new(CommandBufferInner { vk, resources: vec![] }) })
	}
}
impl Drop for CommandBuffer {
	fn drop(&mut self) {
		self.pool.free.lock().unwrap().push(self.inner.get_mut().unwrap().vk);
	}
}

pub(crate) struct CommandBufferInner {
	pub(crate) vk: vk::CommandBuffer,
	resources: Vec<Arc<dyn BufferAbstract>>,
}

pub struct CommandBufferRecording<'a> {
	device: &'a Device,
	inner: &'a mut CommandBufferInner,
}
impl<'a> CommandBufferRecording<'a> {
	pub fn copy_buffer<T: ?Sized + 'static>(&mut self, src: Arc<Buffer<T>>, dst: Arc<Buffer<T>>) {
		assert!(src.size() <= dst.size());

		let regions = [vk::BufferCopy::builder().size(src.size()).build()];
		unsafe { self.device.vk.cmd_copy_buffer(self.inner.vk, src.vk, dst.vk, &regions) };

		self.inner.resources.push(src);
		self.inner.resources.push(dst);
	}
}
