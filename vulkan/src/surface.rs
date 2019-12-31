pub use ash::vk::{
	ColorSpaceKHR as ColorSpace, PresentModeKHR as PresentMode, SurfaceCapabilitiesKHR as SurfaceCapabilities,
	SurfaceFormatKHR as SurfaceFormat, SurfaceTransformFlagsKHR as SurfaceTransformFlags,
};

use crate::instance::Instance;
use ash::vk;
use std::sync::Arc;

pub struct Surface<T> {
	instance: Arc<Instance>,
	window: T,
	pub vk: vk::SurfaceKHR,
}
impl<T> Surface<T> {
	pub fn window(&self) -> &T {
		&self.window
	}

	pub(crate) unsafe fn from_vk(instance: Arc<Instance>, window: T, vk: vk::SurfaceKHR) -> Arc<Self> {
		Arc::new(Self { instance, window, vk })
	}
}
impl<T> Drop for Surface<T> {
	fn drop(&mut self) {
		unsafe { self.instance.khr_surface.destroy_surface(self.vk, None) };
	}
}
