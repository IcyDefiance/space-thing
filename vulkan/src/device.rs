use crate::{image::ImageAbstract, pipeline::PipelineBuilder, render_pass::RenderPass};
pub use ash::vk::BufferUsageFlags;

use crate::{
	buffer::BufferInit,
	command::{CommandBuffer, CommandPool},
	image::{Format, Framebuffer, ImageSubresourceRange, ImageView},
	instance::Instance,
	physical_device::{PhysicalDevice, QueueFamily},
	pipeline::PipelineLayout,
	shader::ShaderModule,
	surface::{ColorSpace, PresentMode, Surface, SurfaceTransformFlags},
	swapchain::{CompositeAlphaFlags, Swapchain, SwapchainImage},
	sync::Fence,
	Extent2D,
};
use ash::{extensions::khr, version::DeviceV1_0, vk, Device as VkDevice};
use std::{mem::size_of, sync::Arc};
use typenum::Bit;
use vk_mem::{AllocationCreateInfo, Allocator, AllocatorCreateInfo, MemoryUsage};

pub struct Device {
	instance: Arc<Instance>,
	physical_device: vk::PhysicalDevice,
	pub vk: VkDevice,
	pub khr_swapchain: khr::Swapchain,
	pub allocator: Allocator,
}
impl Device {
	pub fn build_pipeline(
		self: &Arc<Self>,
		layout: Arc<PipelineLayout>,
		render_pass: Arc<RenderPass>,
	) -> PipelineBuilder<'static, ()> {
		PipelineBuilder::new(self.clone(), layout, render_pass)
	}

	pub fn create_buffer_slice<T, CPU: Bit>(
		self: &Arc<Self>,
		len: usize,
		_: CPU,
		usage: BufferUsageFlags,
	) -> BufferInit<[T], CPU> {
		let size = size_of::<T>() as u64 * len as u64;

		let ci = ash::vk::BufferCreateInfo::builder().size(size).usage(usage).build();

		let usage = if CPU::BOOL { MemoryUsage::CpuOnly } else { MemoryUsage::GpuOnly };
		let aci = AllocationCreateInfo { usage, ..Default::default() };

		let (vk, alloc, _) = self.allocator.create_buffer(&ci, &aci).unwrap();

		BufferInit::from_vk(self.clone(), vk, alloc, size)
	}

	pub fn create_command_pool<'a>(self: &Arc<Self>, family: QueueFamily<'a>, transient: bool) -> Arc<CommandPool> {
		let mut flags = vk::CommandPoolCreateFlags::empty();
		if transient {
			flags |= vk::CommandPoolCreateFlags::TRANSIENT;
		};

		let ci = vk::CommandPoolCreateInfo::builder().flags(flags).queue_family_index(family.idx);

		let vk = unsafe { self.vk.create_command_pool(&ci, None) }.unwrap();
		unsafe { CommandPool::from_vk(self.clone(), family.idx, vk) }
	}

	pub(crate) fn create_fence(self: &Arc<Self>, signalled: bool, resources: Vec<Arc<CommandBuffer>>) -> Fence {
		unsafe {
			let mut flags = vk::FenceCreateFlags::empty();
			if signalled {
				flags |= vk::FenceCreateFlags::SIGNALED;
			}

			let vk = self.vk.create_fence(&vk::FenceCreateInfo::builder().flags(flags), None).unwrap();
			Fence::from_vk(self.clone(), vk, resources)
		}
	}

	pub fn create_framebuffer(
		self: &Arc<Self>,
		render_pass: Arc<RenderPass>,
		attachments: Vec<Arc<ImageView>>,
		width: u32,
		height: u32,
	) -> Arc<Framebuffer> {
		let attachment_vks: Vec<_> = attachments.iter().map(|x| x.vk).collect();

		let ci = vk::FramebufferCreateInfo::builder()
			.render_pass(render_pass.vk)
			.attachments(&attachment_vks)
			.width(width)
			.height(height)
			.layers(1);
		let vk = unsafe { self.vk.create_framebuffer(&ci, None) }.unwrap();
		unsafe { Framebuffer::from_vk(render_pass, attachments, vk) }
	}

	pub fn create_image_view(
		&self,
		image: Arc<dyn ImageAbstract>,
		format: Format,
		subresource_range: ImageSubresourceRange,
	) -> Arc<ImageView> {
		let ci = vk::ImageViewCreateInfo::builder()
			.image(image.vk())
			.view_type(vk::ImageViewType::TYPE_2D)
			.format(format)
			.subresource_range(subresource_range);
		let vk = unsafe { self.vk.create_image_view(&ci, None) }.unwrap();
		unsafe { ImageView::from_vk(image, vk) }
	}

	pub fn create_pipeline_layout(self: &Arc<Self>) -> Arc<PipelineLayout> {
		let ci = vk::PipelineLayoutCreateInfo::builder();
		let vk = unsafe { self.vk.create_pipeline_layout(&ci, None) }.unwrap();
		unsafe { PipelineLayout::from_vk(self.clone(), vk) }
	}

	pub unsafe fn create_shader_module(self: &Arc<Self>, code: &[u32]) -> Arc<ShaderModule> {
		let ci = vk::ShaderModuleCreateInfo::builder().code(code);
		let vk = self.vk.create_shader_module(&ci, None).unwrap();
		ShaderModule::from_vk(self.clone(), vk)
	}

	pub fn create_swapchain<'a, T>(
		self: &Arc<Self>,
		surface: Arc<Surface<T>>,
		min_image_count: u32,
		image_format: Format,
		image_color_space: ColorSpace,
		image_extent: Extent2D,
		queue_families: impl IntoIterator<Item = QueueFamily<'a>>,
		pre_transform: SurfaceTransformFlags,
		composite_alpha: CompositeAlphaFlags,
		present_mode: PresentMode,
		old_swapchain: Option<&Swapchain<T>>,
	) -> (Arc<Swapchain<T>>, impl Iterator<Item = Arc<SwapchainImage<T>>>) {
		let queue_family_indices: Vec<_> = queue_families
			.into_iter()
			.inspect(|qfam| assert!(self.physical_device() == qfam.physical_device()))
			.map(|qfam| qfam.idx)
			.collect();

		let image_sharing_mode =
			if queue_family_indices.len() > 1 { vk::SharingMode::CONCURRENT } else { vk::SharingMode::EXCLUSIVE };

		let ci = vk::SwapchainCreateInfoKHR::builder()
			.surface(surface.vk)
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
			.old_swapchain(old_swapchain.map(|x| x.vk).unwrap_or(vk::SwapchainKHR::null()));
		let vk = unsafe { self.khr_swapchain.create_swapchain(&ci, None) }.unwrap();
		let swapchain = unsafe { Swapchain::from_vk(self.clone(), surface, vk) };

		let swapchain2 = swapchain.clone();
		let images = unsafe { self.khr_swapchain.get_swapchain_images(swapchain.vk) }
			.unwrap()
			.into_iter()
			.map(move |vk| unsafe { SwapchainImage::from_vk(swapchain2.clone(), vk) });

		(swapchain, images)
	}

	pub fn physical_device(&self) -> PhysicalDevice {
		PhysicalDevice::from_vk(&self.instance, self.physical_device)
	}

	pub(crate) fn from_vk(instance: Arc<Instance>, physical_device: vk::PhysicalDevice, vk: VkDevice) -> Arc<Self> {
		let khr_swapchain = khr::Swapchain::new(&instance.vk, &vk);

		let ci = AllocatorCreateInfo {
			physical_device,
			device: vk.clone(),
			instance: instance.vk.clone(),
			..AllocatorCreateInfo::default()
		};
		let allocator = Allocator::new(&ci).unwrap();

		Arc::new(Self { instance, physical_device, vk, khr_swapchain, allocator })
	}

	pub(crate) unsafe fn get_queue(self: &Arc<Self>, queue_family_index: u32, queue_index: u32) -> Arc<Queue> {
		let vk = self.vk.get_device_queue(queue_family_index, queue_index);

		Arc::new(Queue { device: self.clone(), family: queue_family_index, vk })
	}
}
impl Drop for Device {
	fn drop(&mut self) {
		self.allocator.destroy();
		unsafe { self.vk.destroy_device(None) };
	}
}

pub struct Queue {
	device: Arc<Device>,
	family: u32,
	pub vk: vk::Queue,
}
impl Queue {
	pub fn device(&self) -> &Arc<Device> {
		&self.device
	}

	pub fn family(&self) -> QueueFamily {
		QueueFamily::from_vk(self.device.physical_device(), self.family)
	}

	pub fn submit(self: &Arc<Self>, cmd: Arc<CommandBuffer>) -> SubmitFuture {
		assert!(cmd.pool.queue_family == self.family);

		SubmitFuture { queue: self.clone(), cmd }
	}
}

pub struct SubmitFuture {
	queue: Arc<Queue>,
	cmd: Arc<CommandBuffer>,
}
impl SubmitFuture {
	pub fn end(self) -> Fence {
		let fence = self.queue.device.create_fence(false, vec![self.cmd.clone()]);

		let cmd_inner = self.cmd.inner.read().unwrap();
		let submits = [vk::SubmitInfo::builder().command_buffers(&[cmd_inner.vk]).build()];
		unsafe { self.queue.device().vk.queue_submit(self.queue.vk, &submits, fence.vk) }.unwrap();

		fence
	}
}
