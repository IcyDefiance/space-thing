use crate::{image::ImageAbstract, physical_device::QueueFamily};
pub use ash::vk::CompositeAlphaFlagsKHR as CompositeAlphaFlags;

use crate::{
	device::Device,
	image::Format,
	surface::{ColorSpace, PresentMode, Surface, SurfaceTransformFlags},
	Extent2D,
};
use ash::vk;
use std::sync::Arc;

pub struct Swapchain<T> {
	device: Arc<Device>,
	surface: Arc<Surface<T>>,
	pub vk: vk::SwapchainKHR,
}
impl<T> Swapchain<T> {
	pub fn recreate<'a>(
		&self,
		min_image_count: u32,
		image_format: Format,
		image_color_space: ColorSpace,
		image_extent: Extent2D,
		queue_families: impl IntoIterator<Item = QueueFamily<'a>>,
		pre_transform: SurfaceTransformFlags,
		composite_alpha: CompositeAlphaFlags,
		present_mode: PresentMode,
	) -> (Arc<Swapchain<T>>, impl Iterator<Item = Arc<SwapchainImage<T>>>) {
		let queue_family_indices: Vec<_> = queue_families
			.into_iter()
			.inspect(|qfam| assert!(self.device.physical_device() == qfam.physical_device()))
			.map(|qfam| qfam.idx)
			.collect();

		let image_sharing_mode =
			if queue_family_indices.len() > 1 { vk::SharingMode::CONCURRENT } else { vk::SharingMode::EXCLUSIVE };

		let ci = vk::SwapchainCreateInfoKHR::builder()
			.surface(self.surface.vk)
			.min_image_count(min_image_count)
			.image_format(image_format)
			.image_color_space(image_color_space)
			.image_extent(image_extent)
			.image_array_layers(1)
			.image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
			.image_sharing_mode(image_sharing_mode)
			.queue_family_indices(&queue_family_indices)
			.pre_transform(pre_transform)
			.composite_alpha(composite_alpha)
			.present_mode(present_mode)
			.clipped(true)
			.old_swapchain(self.vk);
		let vk = unsafe { self.device.khr_swapchain.create_swapchain(&ci, None) }.unwrap();
		let swapchain = unsafe { Swapchain::from_vk(self.device.clone(), self.surface.clone(), vk) };

		let swapchain2 = swapchain.clone();
		let images = unsafe { self.device.khr_swapchain.get_swapchain_images(swapchain.vk) }
			.unwrap()
			.into_iter()
			.map(move |vk| unsafe { SwapchainImage::from_vk(swapchain2.clone(), vk) });

		(swapchain, images)
	}

	pub fn surface(&self) -> &Arc<Surface<T>> {
		&self.surface
	}

	pub(crate) unsafe fn from_vk(device: Arc<Device>, surface: Arc<Surface<T>>, vk: vk::SwapchainKHR) -> Arc<Self> {
		Arc::new(Self { device, surface, vk })
	}
}
impl<T> Drop for Swapchain<T> {
	fn drop(&mut self) {
		unsafe { self.device.khr_swapchain.destroy_swapchain(self.vk, None) };
	}
}

pub struct SwapchainImage<T> {
	swapchain: Arc<Swapchain<T>>,
	vk: vk::Image,
}
impl<T> SwapchainImage<T> {
	pub(crate) unsafe fn from_vk(swapchain: Arc<Swapchain<T>>, vk: vk::Image) -> Arc<Self> {
		Arc::new(Self { swapchain, vk })
	}
}
impl<T> ImageAbstract for SwapchainImage<T> {
	fn device(&self) -> &Arc<Device> {
		&self.swapchain.device
	}

	fn vk(&self) -> vk::Image {
		self.vk
	}
}
