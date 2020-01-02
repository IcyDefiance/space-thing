use crate::{
	device::Queue,
	image::ImageAbstract,
	physical_device::QueueFamily,
	sync::{GpuFuture, Semaphore},
};
pub use ash::vk::CompositeAlphaFlagsKHR as CompositeAlphaFlags;
use std::sync::Mutex;

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
	semaphores: Mutex<Vec<Vec<Arc<Semaphore>>>>,
}
impl<T> Swapchain<T> {
	pub fn acquire_next_image(self: &Arc<Self>, timeout: u64) -> Result<(u32, bool, AcquireFuture<T>), vk::Result> {
		let semaphore = self.device.create_semaphore();
		let (image_idx, suboptimal) =
			unsafe { self.device.khr_swapchain.acquire_next_image(self.vk, timeout, semaphore.vk, vk::Fence::null()) }?;
		Ok((image_idx, suboptimal, AcquireFuture { _swapchain: self.clone(), semaphore }))
	}

	pub fn present_after(
		prev: impl GpuFuture,
		queue: Arc<Queue>,
		swapchains: &[Arc<Swapchain<T>>],
		image_indices: &[u32],
	) -> Result<bool, vk::Result> {
		let semaphores = prev.semaphores().0;
		let semaphore_vks: Vec<_> = semaphores.iter().map(|x| x.vk).collect();

		let swapchain_vks: Vec<_> = swapchains.iter().map(|x| x.vk).collect();

		for (swapchain, &idx) in swapchains.iter().zip(image_indices) {
			swapchain.semaphores.lock().unwrap()[idx as usize] = semaphores.clone();
		}

		// TODO: check individual results
		let ci = vk::PresentInfoKHR::builder()
			.wait_semaphores(&semaphore_vks)
			.swapchains(&swapchain_vks)
			.image_indices(image_indices);
		let suboptimal = unsafe { queue.device.khr_swapchain.queue_present(queue.vk, &ci) }?;

		Ok(suboptimal)
	}

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
		let images = unsafe { self.device.khr_swapchain.get_swapchain_images(vk) }.unwrap();

		let swapchain = unsafe { Swapchain::from_vk(self.device.clone(), self.surface.clone(), vk, images.len()) };
		let swapchain2 = swapchain.clone();
		let images = images.into_iter().map(move |vk| unsafe { SwapchainImage::from_vk(swapchain2.clone(), vk) });

		(swapchain, images)
	}

	pub fn surface(&self) -> &Arc<Surface<T>> {
		&self.surface
	}

	pub(crate) unsafe fn from_vk(
		device: Arc<Device>,
		surface: Arc<Surface<T>>,
		vk: vk::SwapchainKHR,
		image_count: usize,
	) -> Arc<Self> {
		let semaphores = Mutex::new((0..image_count).map(|_| vec![]).collect());
		Arc::new(Self { device, surface, vk, semaphores })
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

pub struct AcquireFuture<T> {
	_swapchain: Arc<Swapchain<T>>,
	semaphore: Arc<Semaphore>,
}
impl<T> GpuFuture for AcquireFuture<T> {
	fn semaphores(self) -> (Vec<Arc<Semaphore>>, Vec<vk::PipelineStageFlags>) {
		(vec![self.semaphore], vec![vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
	}
}
