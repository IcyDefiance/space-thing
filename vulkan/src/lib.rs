pub mod buffer;
pub mod command;
pub mod device;
pub mod image;
pub mod instance;
pub mod physical_device;
pub mod pipeline;
pub mod render_pass;
pub mod shader;
pub mod surface;
pub mod swapchain;
pub mod sync;

pub use ash::{
	vk::{Extent2D, Offset2D, Rect2D},
	LoadingError,
};

use ash::Entry;
use std::sync::Arc;

pub struct Vulkan {
	pub vk: Entry,
}
impl Vulkan {
	pub fn new() -> Result<Arc<Self>, LoadingError> {
		Ok(Arc::new(Self { vk: Entry::new()? }))
	}
}

pub trait Owned<T> {
	fn inner(&self) -> &T;
}
impl<T> Owned<T> for T {
	fn inner(&self) -> &T {
		self
	}
}
impl<T> Owned<T> for Arc<T> {
	fn inner(&self) -> &T {
		self
	}
}
