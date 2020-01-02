use crate::{
	buffer::BufferAbstract, command::CommandBuffer, device::Device, image::Framebuffer, pipeline::Pipeline,
	render_pass::RenderPass,
};
use ash::{version::DeviceV1_0, vk};
use std::sync::{Arc, Mutex};
use typenum::{B0, B1};

pub struct Fence {
	device: Arc<Device>,
	pub vk: vk::Fence,
	pub(crate) resources: Mutex<Vec<Arc<CommandBuffer<B0>>>>,
}
impl Fence {
	pub fn wait(&self) {
		unsafe { self.device.vk.wait_for_fences(&[self.vk], false, !0) }.unwrap();
		self.resources.lock().unwrap().clear();
	}

	pub(crate) unsafe fn from_vk(device: Arc<Device>, vk: vk::Fence, resources: Vec<Arc<CommandBuffer<B0>>>) -> Self {
		Self { device, vk, resources: Mutex::new(resources) }
	}
}
impl Drop for Fence {
	fn drop(&mut self) {
		self.wait();
		unsafe { self.device.vk.destroy_fence(self.vk, None) };
	}
}

pub struct Semaphore {
	pub(crate) device: Arc<Device>,
	pub vk: vk::Semaphore,
}
impl Semaphore {
	pub(crate) unsafe fn from_vk(device: Arc<Device>, vk: vk::Semaphore) -> Arc<Self> {
		Arc::new(Self { device, vk })
	}
}
impl Drop for Semaphore {
	fn drop(&mut self) {
		unsafe { self.device.vk.destroy_semaphore(self.vk, None) };
	}
}

pub trait GpuFuture {
	fn semaphores(self) -> (Vec<Arc<Semaphore>>, Vec<vk::PipelineStageFlags>);
}

pub(crate) enum Resource {
	Buffer(Arc<dyn BufferAbstract>),
	CommandBuffer(Arc<CommandBuffer<B1>>),
	Framebuffer(Arc<Framebuffer>),
	Pipeline(Arc<Pipeline>),
	RenderPass(Arc<RenderPass>),
	Semaphore(Arc<Semaphore>),
}
