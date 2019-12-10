use crate::gfx::Gfx;
use ash::{version::DeviceV1_0, vk};
use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use std::{
	cmp::{max, min},
	ffi::CStr,
	slice,
	sync::Arc,
	u32, u64,
};
use winit::{EventsLoop, WindowBuilder};

pub struct Window {
	gfx: Arc<Gfx>,
	window: winit::Window,
	surface: vk::SurfaceKHR,
	swapchain: vk::SwapchainKHR,
	image_views: Vec<vk::ImageView>,
	layout: vk::PipelineLayout,
	render_pass: vk::RenderPass,
	pipeline: vk::Pipeline,
	framebuffers: Vec<vk::Framebuffer>,
	command_pool: vk::CommandPool,
	command_buffers: Vec<vk::CommandBuffer>,
	image_available: Vec<vk::Semaphore>,
	render_finished: Vec<vk::Semaphore>,
	frame_finished: Vec<vk::Fence>,
	frame: usize,
}
impl Window {
	pub fn new(gfx: &Arc<Gfx>, events_loop: &EventsLoop) -> Self {
		let window = WindowBuilder::new().with_dimensions((1440, 810).into()).build(&events_loop).unwrap();

		let surface = match window.raw_window_handle() {
			#[cfg(windows)]
			RawWindowHandle::Windows(handle) => {
				let ci = vk::Win32SurfaceCreateInfoKHR::builder().hinstance(handle.hinstance).hwnd(handle.hwnd);
				unsafe { gfx.khr_win32_surface.create_win32_surface(&ci, None) }.unwrap()
			},
			#[cfg(unix)]
			RawWindowHandle::Xlib(handle) => {
				let ci = vk::XlibSurfaceCreateInfoKHR::builder().dpy(handle.display as _).window(handle.window);
				unsafe { gfx.khr_xlib_surface.create_xlib_surface(&ci, None) }.unwrap()
			},
			#[cfg(unix)]
			RawWindowHandle::Wayland(handle) => {
				let ci = vk::WaylandSurfaceCreateInfoKHR::builder().display(handle.display).surface(handle.surface);
				unsafe { gfx.khr_wayland_surface.create_wayland_surface(&ci, None) }.unwrap()
			},
			_ => unimplemented!(),
		};
		assert!(unsafe {
			gfx.khr_surface.get_physical_device_surface_support(gfx.physical_device, gfx.queue_family, surface)
		});

		let caps =
			unsafe { gfx.khr_surface.get_physical_device_surface_capabilities(gfx.physical_device, surface) }.unwrap();
		let surface_format =
			unsafe { gfx.khr_surface.get_physical_device_surface_formats(gfx.physical_device, surface) }
				.unwrap()
				.into_iter()
				.max_by_key(|format| {
					format.format == vk::Format::B8G8R8A8_UNORM
						&& format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
				})
				.unwrap();
		let image_extent = if caps.current_extent.width != u32::MAX {
			caps.current_extent
		} else {
			let (width, height) = window.get_inner_size().unwrap().to_physical(1.0).into();
			vk::Extent2D {
				width: max(caps.min_image_extent.width, min(caps.max_image_extent.width, width)),
				height: max(caps.min_image_extent.height, min(caps.max_image_extent.height, height)),
			}
		};
		let queue_family_indices = [gfx.queue_family];
		let present_mode =
			unsafe { gfx.khr_surface.get_physical_device_surface_present_modes(gfx.physical_device, surface) }
				.unwrap()
				.into_iter()
				.min_by_key(|&mode| match mode {
					vk::PresentModeKHR::MAILBOX => 0,
					vk::PresentModeKHR::IMMEDIATE => 1,
					vk::PresentModeKHR::FIFO_RELAXED => 2,
					vk::PresentModeKHR::FIFO => 3,
					_ => 4,
				})
				.unwrap();
		let ci = vk::SwapchainCreateInfoKHR::builder()
			.surface(surface)
			.min_image_count(caps.min_image_count + 1)
			.image_format(surface_format.format)
			.image_color_space(surface_format.color_space)
			.image_extent(image_extent)
			.image_array_layers(1)
			.image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
			.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
			.queue_family_indices(&queue_family_indices)
			.pre_transform(caps.current_transform)
			.composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
			.present_mode(present_mode)
			.clipped(true);
		let swapchain = unsafe { gfx.khr_swapchain.create_swapchain(&ci, None) }.unwrap();

		let image_views: Vec<_> = unsafe { gfx.khr_swapchain.get_swapchain_images(swapchain) }
			.unwrap()
			.into_iter()
			.map(|image| {
				let ci = vk::ImageViewCreateInfo::builder()
					.image(image)
					.view_type(vk::ImageViewType::TYPE_2D)
					.format(surface_format.format)
					.subresource_range(
						vk::ImageSubresourceRange::builder()
							.aspect_mask(vk::ImageAspectFlags::COLOR)
							.level_count(1)
							.layer_count(1)
							.build(),
					);
				unsafe { gfx.device.create_image_view(&ci, None) }.unwrap()
			})
			.collect();

		let ci = vk::PipelineLayoutCreateInfo::builder();
		let layout = unsafe { gfx.device.create_pipeline_layout(&ci, None) }.unwrap();

		let attachments = [vk::AttachmentDescription::builder()
			.format(surface_format.format)
			.samples(vk::SampleCountFlags::TYPE_1)
			.load_op(vk::AttachmentLoadOp::CLEAR)
			.store_op(vk::AttachmentStoreOp::STORE)
			.stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
			.stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
			.initial_layout(vk::ImageLayout::UNDEFINED)
			.final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
			.build()];
		let color_attachments =
			[vk::AttachmentReference::builder().layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL).build()];
		let subpasses = [vk::SubpassDescription::builder()
			.pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
			.color_attachments(&color_attachments)
			.build()];
		let dependencies = [vk::SubpassDependency::builder()
			.src_subpass(vk::SUBPASS_EXTERNAL)
			.src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
			.dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
			.dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
			.build()];
		let ci = vk::RenderPassCreateInfo::builder()
			.attachments(&attachments)
			.subpasses(&subpasses)
			.dependencies(&dependencies);
		let render_pass = unsafe { gfx.device.create_render_pass(&ci, None) }.unwrap();

		let stages = [
			vk::PipelineShaderStageCreateInfo::builder()
				.stage(vk::ShaderStageFlags::VERTEX)
				.module(gfx.vshader)
				.name(CStr::from_bytes_with_nul(b"main\0").unwrap())
				.build(),
			vk::PipelineShaderStageCreateInfo::builder()
				.stage(vk::ShaderStageFlags::FRAGMENT)
				.module(gfx.fshader)
				.name(CStr::from_bytes_with_nul(b"main\0").unwrap())
				.build(),
		];
		let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder();
		let input_assembly_state =
			vk::PipelineInputAssemblyStateCreateInfo::builder().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
		let viewports = [vk::Viewport::builder()
			.width(image_extent.width as _)
			.height(image_extent.height as _)
			.max_depth(1.0)
			.build()];
		let scissors = [vk::Rect2D::builder().extent(image_extent).build()];
		let viewport_state = vk::PipelineViewportStateCreateInfo::builder().viewports(&viewports).scissors(&scissors);
		let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
			.polygon_mode(vk::PolygonMode::FILL)
			.cull_mode(vk::CullModeFlags::BACK)
			.front_face(vk::FrontFace::CLOCKWISE)
			.line_width(1.0);
		let multisample_state =
			vk::PipelineMultisampleStateCreateInfo::builder().rasterization_samples(vk::SampleCountFlags::TYPE_1);
		let attachments =
			[vk::PipelineColorBlendAttachmentState::builder().color_write_mask(vk::ColorComponentFlags::all()).build()];
		let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder().attachments(&attachments);
		let cis = [vk::GraphicsPipelineCreateInfo::builder()
			.stages(&stages)
			.vertex_input_state(&vertex_input_state)
			.input_assembly_state(&input_assembly_state)
			.viewport_state(&viewport_state)
			.rasterization_state(&rasterization_state)
			.multisample_state(&multisample_state)
			.color_blend_state(&color_blend_state)
			.layout(layout)
			.render_pass(render_pass)
			.build()];
		let pipeline =
			unsafe { gfx.device.create_graphics_pipelines(vk::PipelineCache::null(), &cis, None) }.unwrap()[0];

		let framebuffers: Vec<_> = image_views
			.iter()
			.map(|view| {
				let ci = vk::FramebufferCreateInfo::builder()
					.render_pass(render_pass)
					.attachments(slice::from_ref(view))
					.width(image_extent.width)
					.height(image_extent.height)
					.layers(1);
				unsafe { gfx.device.create_framebuffer(&ci, None) }.unwrap()
			})
			.collect();

		let ci = vk::CommandPoolCreateInfo::builder().queue_family_index(gfx.queue_family);
		let command_pool = unsafe { gfx.device.create_command_pool(&ci, None) }.unwrap();

		let ci = vk::CommandBufferAllocateInfo::builder()
			.command_pool(command_pool)
			.level(vk::CommandBufferLevel::PRIMARY)
			.command_buffer_count(framebuffers.len() as _);
		let command_buffers = unsafe { gfx.device.allocate_command_buffers(&ci) }.unwrap();
		for (&cmd, &framebuffer) in command_buffers.iter().zip(&framebuffers) {
			unsafe {
				gfx.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder()).unwrap();

				let ci = vk::RenderPassBeginInfo::builder()
					.render_pass(render_pass)
					.framebuffer(framebuffer)
					.render_area(vk::Rect2D::builder().extent(image_extent).build())
					.clear_values(&[vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } }]);
				gfx.device.cmd_begin_render_pass(cmd, &ci, vk::SubpassContents::INLINE);
				gfx.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
				gfx.device.cmd_draw(cmd, 3, 1, 0, 0);
				gfx.device.cmd_end_render_pass(cmd);

				gfx.device.end_command_buffer(cmd).unwrap();
			}
		}

		let image_available = framebuffers
			.iter()
			.map(|_| unsafe { gfx.device.create_semaphore(&vk::SemaphoreCreateInfo::builder(), None) }.unwrap())
			.collect();
		let render_finished = framebuffers
			.iter()
			.map(|_| unsafe { gfx.device.create_semaphore(&vk::SemaphoreCreateInfo::builder(), None) }.unwrap())
			.collect();
		let frame_finished = framebuffers
			.iter()
			.map(|_| {
				unsafe {
					gfx.device.create_fence(&vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED), None)
				}
				.unwrap()
			})
			.collect();

		Self {
			gfx: gfx.clone(),
			window,
			surface,
			swapchain,
			image_views,
			layout,
			render_pass,
			pipeline,
			framebuffers,
			command_pool,
			command_buffers,
			image_available,
			render_finished,
			frame_finished,
			frame: 0,
		}
	}

	pub fn draw(&mut self) {
		unsafe { self.gfx.device.wait_for_fences(&self.frame_finished[self.frame..self.frame + 1], false, u64::MAX) }
			.unwrap();
		unsafe { self.gfx.device.reset_fences(&self.frame_finished[self.frame..self.frame + 1]) }.unwrap();

		let (i, _suboptimal) = unsafe {
			self.gfx.khr_swapchain.acquire_next_image(
				self.swapchain,
				u64::MAX,
				self.image_available[self.frame],
				vk::Fence::null(),
			)
		}
		.unwrap();

		let submits = [vk::SubmitInfo::builder()
			.wait_semaphores(&[self.image_available[self.frame]])
			.wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
			.command_buffers(&self.command_buffers[(i as usize)..(i as usize) + 1])
			.signal_semaphores(&[self.render_finished[self.frame]])
			.build()];
		unsafe { self.gfx.device.queue_submit(self.gfx.queue, &submits, self.frame_finished[self.frame]) }.unwrap();

		let ci = vk::PresentInfoKHR::builder()
			.wait_semaphores(slice::from_ref(&self.render_finished[self.frame]))
			.swapchains(slice::from_ref(&self.swapchain))
			.image_indices(slice::from_ref(&i));
		let _suboptimal = unsafe { self.gfx.khr_swapchain.queue_present(self.gfx.queue, &ci) }.unwrap();

		self.frame = (self.frame + 1) % 2;
	}
}
impl Drop for Window {
	fn drop(&mut self) {
		unsafe {
			// this is equivalent to `(self.frame - 1) mod 2` and avoids unsigned integer overflow
			let frame = (self.frame + 1) % 2;
			self.gfx.device.wait_for_fences(&self.frame_finished[frame..frame + 1], false, u64::MAX).unwrap();
			for &fence in &self.frame_finished {
				self.gfx.device.destroy_fence(fence, None);
			}
			for &semaphore in &self.render_finished {
				self.gfx.device.destroy_semaphore(semaphore, None);
			}
			for &semaphore in &self.image_available {
				self.gfx.device.destroy_semaphore(semaphore, None);
			}
			self.gfx.device.free_command_buffers(self.command_pool, &self.command_buffers);
			self.gfx.device.destroy_command_pool(self.command_pool, None);
			for &framebuffer in &self.framebuffers {
				self.gfx.device.destroy_framebuffer(framebuffer, None);
			}
			self.gfx.device.destroy_pipeline(self.pipeline, None);
			self.gfx.device.destroy_render_pass(self.render_pass, None);
			self.gfx.device.destroy_pipeline_layout(self.layout, None);
			for &image_view in &self.image_views {
				self.gfx.device.destroy_image_view(image_view, None);
			}
			self.gfx.khr_swapchain.destroy_swapchain(self.swapchain, None);
			self.gfx.khr_surface.destroy_surface(self.surface, None);
		}
	}
}
