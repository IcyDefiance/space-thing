pub use ash::vk::{Format, ImageSubresourceRange};

use crate::{device::Device, render_pass::RenderPass};
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

pub struct Framebuffer {
	render_pass: Arc<RenderPass>,
	_attachments: Vec<Arc<ImageView>>,
	pub vk: vk::Framebuffer,
}
impl Framebuffer {
	pub(crate) unsafe fn from_vk(
		render_pass: Arc<RenderPass>,
		attachments: Vec<Arc<ImageView>>,
		vk: vk::Framebuffer,
	) -> Arc<Self> {
		Arc::new(Self { render_pass, _attachments: attachments, vk })
	}
}
impl Drop for Framebuffer {
	fn drop(&mut self) {
		unsafe { self.render_pass.device().vk.destroy_framebuffer(self.vk, None) };
	}
}

pub struct ImageView {
	image: Arc<dyn ImageAbstract>,
	pub vk: vk::ImageView,
}
impl ImageView {
	pub(crate) unsafe fn from_vk(image: Arc<dyn ImageAbstract>, vk: vk::ImageView) -> Arc<Self> {
		Arc::new(Self { image, vk })
	}
}
impl Drop for ImageView {
	fn drop(&mut self) {
		unsafe { self.image.device().vk.destroy_image_view(self.vk, None) };
	}
}

pub trait ImageAbstract {
	fn device(&self) -> &Arc<Device>;
	fn vk(&self) -> vk::Image;
}
