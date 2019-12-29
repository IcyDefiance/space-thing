use crate::device::Device;
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

pub struct PipelineLayout {
	device: Arc<Device>,
	pub vk: vk::PipelineLayout,
}
impl PipelineLayout {
	pub(crate) fn from_vk(device: Arc<Device>, vk: vk::PipelineLayout) -> Self {
		Self { device, vk }
	}
}
impl Drop for PipelineLayout {
	fn drop(&mut self) {
		unsafe { self.device.vk.destroy_pipeline_layout(self.vk, None) };
	}
}
