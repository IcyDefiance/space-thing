use crate::device::Device;
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

pub struct ShaderModule {
	device: Arc<Device>,
	pub vk: vk::ShaderModule,
}
impl ShaderModule {
	pub(crate) fn from_vk(device: Arc<Device>, vk: vk::ShaderModule) -> Self {
		Self { device, vk }
	}
}
impl Drop for ShaderModule {
	fn drop(&mut self) {
		unsafe { self.device.vk.destroy_shader_module(self.vk, None) };
	}
}
