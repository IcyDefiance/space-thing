use crate::{command::CommandBuffer, device::Device};
use ash::{version::DeviceV1_0, vk};
use std::sync::{Arc, Mutex};

pub struct Fence {
	device: Arc<Device>,
	pub vk: vk::Fence,
	pub(crate) resources: Mutex<Vec<Arc<CommandBuffer>>>,
}
impl Fence {
	pub fn wait(&self) {
		unsafe { self.device.vk.wait_for_fences(&[self.vk], false, !0) }.unwrap();
		self.resources.lock().unwrap().clear();
	}

	pub(crate) unsafe fn from_vk(device: Arc<Device>, vk: vk::Fence, resources: Vec<Arc<CommandBuffer>>) -> Self {
		Self { device, vk, resources: Mutex::new(resources) }
	}
}
impl Drop for Fence {
	fn drop(&mut self) {
		self.wait();
		unsafe { self.device.vk.destroy_fence(self.vk, None) };
	}
}
