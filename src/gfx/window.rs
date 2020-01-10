use crate::gfx::{camera::Camera, image::transition_layout, world::World, Gfx, TriangleVertex};
use ash::{version::DeviceV1_0, vk, Device};
use nalgebra::Vector3;
use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use std::{
	cmp::{max, min},
	ffi::CStr,
	mem::size_of,
	slice,
	sync::Arc,
	u32,
};
use winit::{event_loop::EventLoop, window::WindowBuilder};

pub struct Window {
	pub(super) gfx: Arc<Gfx>,
	window: winit::window::Window,
	surface: vk::SurfaceKHR,
	surface_format: vk::SurfaceFormatKHR,
	pub(super) render_pass: vk::RenderPass,
	image_extent: vk::Extent2D,
	swapchain: vk::SwapchainKHR,
	image_views: Vec<vk::ImageView>,
	pub(super) pipeline: vk::Pipeline,
	pub(super) framebuffers: Vec<vk::Framebuffer>,
	stencil_desc_pool: vk::DescriptorPool,
	stencil_desc_sets: Vec<vk::DescriptorSet>,
	frame_data: [FrameData; 2],
	frame: bool,
	recreate_swapchain: bool,
}
impl Window {
	pub fn new(gfx: Arc<Gfx>, event_loop: &EventLoop<()>) -> Self {
		let window = WindowBuilder::new().with_inner_size((1440, 810).into()).build(&event_loop).unwrap();

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

		let surface_format =
			unsafe { gfx.khr_surface.get_physical_device_surface_formats(gfx.physical_device, surface) }
				.unwrap()
				.into_iter()
				.max_by_key(|format| {
					format.format == vk::Format::B8G8R8A8_UNORM
						&& format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
				})
				.unwrap();

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

		let (caps, image_extent) = get_caps(&gfx, surface, &window);
		let (swapchain, image_views) =
			create_swapchain(&gfx, surface, &caps, &surface_format, image_extent, vk::SwapchainKHR::null());
		let pipeline = create_pipeline(&gfx, image_extent, render_pass);
		let framebuffers = create_framebuffers(&gfx, &image_views, render_pass, image_extent);

		let (stencil_desc_pool, stencil_desc_sets) = create_stencil_desc_pool(&gfx, image_views.len() as _);

		let frame_data = [FrameData::new(&gfx), FrameData::new(&gfx)];

		Self {
			gfx,
			window,
			surface,
			surface_format,
			render_pass,
			image_extent,
			swapchain,
			image_views,
			pipeline,
			framebuffers,
			stencil_desc_pool,
			stencil_desc_sets,
			frame_data,
			frame: false,
			recreate_swapchain: false,
		}
	}

	pub fn draw(&mut self, world: &mut World, camera: &Camera) {
		unsafe {
			if self.recreate_swapchain {
				self.recreate_swapchain();
			}

			let frame = self.frame as usize;
			let frame_data = &mut self.frame_data[frame];

			let res = self.gfx.khr_swapchain.acquire_next_image(
				self.swapchain,
				!0,
				frame_data.image_available,
				vk::Fence::null(),
			);
			let image_idx = match res {
				Ok((idx, suboptimal)) => {
					if suboptimal {
						self.recreate_swapchain = true;
					}
					idx
				},
				Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
					self.recreate_swapchain = true;
					return;
				},
				Err(err) => panic!(err),
			};
			let image_uidx = image_idx as usize;

			self.gfx.device.wait_for_fences(&[frame_data.frame_finished], false, !0).unwrap();
			self.gfx.device.reset_fences(&[frame_data.frame_finished]).unwrap();
			self.frame = !self.frame;

			let framebuffer = self.framebuffers[image_uidx];

			self.gfx.device.reset_command_pool(frame_data.cmdpool, vk::CommandPoolResetFlags::empty()).unwrap();

			// TODO: replace with real volumes
			let volumes_len = 1;
			if frame_data.secondaries.len() < volumes_len {
				let ci = vk::CommandBufferAllocateInfo::builder()
					.command_pool(frame_data.cmdpool)
					.level(vk::CommandBufferLevel::SECONDARY)
					.command_buffer_count((volumes_len - frame_data.secondaries.len()) as _);
				frame_data.secondaries.extend(self.gfx.device.allocate_command_buffers(&ci).unwrap());
			}

			for (_, &cmd) in (0..volumes_len).zip(&frame_data.secondaries) {
				let flags =
					vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT | vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE;
				let inherit = vk::CommandBufferInheritanceInfo::builder()
					.render_pass(self.render_pass)
					.subpass(0)
					.framebuffer(framebuffer);
				let info = vk::CommandBufferBeginInfo::builder().flags(flags).inheritance_info(&inherit);
				self.gfx.device.begin_command_buffer(cmd, &info).unwrap();
				self.gfx.device.cmd_push_constants(
					cmd,
					self.gfx.pipeline_layout,
					vk::ShaderStageFlags::FRAGMENT,
					0,
					slice::from_raw_parts(camera as *const _ as _, size_of::<Camera>()),
				);
				self.gfx.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
				self.gfx.device.cmd_bind_vertex_buffers(cmd, 0, &[self.gfx.triangle], &[0]);
				self.gfx.device.cmd_bind_descriptor_sets(
					cmd,
					vk::PipelineBindPoint::GRAPHICS,
					self.gfx.pipeline_layout,
					0,
					&[self.gfx.desc_set, world.desc_set],
					&[],
				);
				self.gfx.device.cmd_draw(cmd, 3, 1, 0, 0);
				self.gfx.device.end_command_buffer(cmd).unwrap();
			}

			let bi = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
			self.gfx.device.begin_command_buffer(frame_data.primary, &bi).unwrap();

			let stencil_desc_set = self.stencil_desc_sets[image_uidx];
			if world.set_cmds.len() > 0 {
				let voxels_out_info = [vk::DescriptorImageInfo::builder()
					.image_view(world.voxels_view)
					.image_layout(vk::ImageLayout::GENERAL)
					.build()];
				let write = [vk::WriteDescriptorSet::builder()
					.dst_set(stencil_desc_set)
					.dst_binding(0)
					.descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
					.image_info(&voxels_out_info)
					.build()];
				self.gfx.device.update_descriptor_sets(&write, &[]);
			}
			for set_cmd in world.set_cmds.drain(..) {
				transition_layout(
					&self.gfx.device,
					frame_data.primary,
					world.voxels,
					1,
					vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
					vk::ImageLayout::GENERAL,
					vk::PipelineStageFlags::FRAGMENT_SHADER,
					vk::PipelineStageFlags::COMPUTE_SHADER,
				);

				self.gfx.device.cmd_push_constants(
					frame_data.primary,
					self.gfx.stencil_pipeline_layout,
					vk::ShaderStageFlags::COMPUTE,
					0,
					slice::from_raw_parts(&set_cmd as *const _ as _, size_of::<Vector3<u32>>()),
				);
				self.gfx.device.cmd_bind_pipeline(
					frame_data.primary,
					vk::PipelineBindPoint::COMPUTE,
					self.gfx.stencil_pipeline,
				);
				self.gfx.device.cmd_bind_descriptor_sets(
					frame_data.primary,
					vk::PipelineBindPoint::COMPUTE,
					self.gfx.stencil_pipeline_layout,
					0,
					&[stencil_desc_set],
					&[],
				);
				self.gfx.device.cmd_dispatch(frame_data.primary, 21, 21, 21);

				transition_layout(
					&self.gfx.device,
					frame_data.primary,
					world.voxels,
					1,
					vk::ImageLayout::GENERAL,
					vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
					vk::PipelineStageFlags::COMPUTE_SHADER,
					vk::PipelineStageFlags::FRAGMENT_SHADER,
				);
			}

			let ci = vk::RenderPassBeginInfo::builder()
				.render_pass(self.render_pass)
				.framebuffer(framebuffer)
				.render_area(vk::Rect2D::builder().extent(self.image_extent).build())
				.clear_values(&[vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } }]);
			self.gfx.device.cmd_begin_render_pass(
				frame_data.primary,
				&ci,
				vk::SubpassContents::SECONDARY_COMMAND_BUFFERS,
			);
			self.gfx.device.cmd_execute_commands(frame_data.primary, &frame_data.secondaries[0..volumes_len]);
			self.gfx.device.cmd_end_render_pass(frame_data.primary);
			self.gfx.device.end_command_buffer(frame_data.primary).unwrap();
			let submits = [vk::SubmitInfo::builder()
				.wait_semaphores(&[frame_data.image_available])
				.wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
				.command_buffers(&[frame_data.primary])
				.signal_semaphores(&[frame_data.render_finished])
				.build()];
			self.gfx.device.queue_submit(self.gfx.queue, &submits, frame_data.frame_finished).unwrap();

			let ci = vk::PresentInfoKHR::builder()
				.wait_semaphores(slice::from_ref(&frame_data.render_finished))
				.swapchains(slice::from_ref(&self.swapchain))
				.image_indices(slice::from_ref(&image_idx));
			match self.gfx.khr_swapchain.queue_present(self.gfx.queue, &ci) {
				Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.recreate_swapchain = true,
				Ok(false) => (),
				Err(err) => panic!(err),
			}
		}
	}

	pub fn winit(&self) -> &winit::window::Window {
		&self.window
	}

	fn recreate_swapchain(&mut self) {
		unsafe {
			self.gfx
				.device
				.wait_for_fences(&[self.frame_data[(!self.frame) as usize].frame_finished], false, !0)
				.unwrap();

			for &framebuffer in &self.framebuffers {
				self.gfx.device.destroy_framebuffer(framebuffer, None);
			}
			self.gfx.device.destroy_pipeline(self.pipeline, None);
			for &image_view in &self.image_views {
				self.gfx.device.destroy_image_view(image_view, None);
			}

			let (caps, image_extent) = get_caps(&self.gfx, self.surface, &self.window);
			let (swapchain, image_views) =
				create_swapchain(&self.gfx, self.surface, &caps, &self.surface_format, image_extent, self.swapchain);
			self.gfx.khr_swapchain.destroy_swapchain(self.swapchain, None);

			if image_views.len() != self.image_views.len() {
				self.gfx.device.destroy_descriptor_pool(self.stencil_desc_pool, None);
				let (stencil_desc_pool, stencil_desc_sets) =
					create_stencil_desc_pool(&self.gfx, image_views.len() as _);
				self.stencil_desc_pool = stencil_desc_pool;
				self.stencil_desc_sets = stencil_desc_sets;
			}

			self.swapchain = swapchain;
			self.image_views = image_views;

			self.pipeline = create_pipeline(&self.gfx, image_extent, self.render_pass);
			self.framebuffers = create_framebuffers(&self.gfx, &self.image_views, self.render_pass, image_extent);

			self.image_extent = image_extent;
		}
	}
}
impl Drop for Window {
	fn drop(&mut self) {
		unsafe {
			self.gfx
				.device
				.wait_for_fences(&[self.frame_data[(!self.frame) as usize].frame_finished], false, !0)
				.unwrap();

			self.frame_data[0].dispose(&self.gfx.device);
			self.frame_data[1].dispose(&self.gfx.device);

			self.gfx.device.destroy_descriptor_pool(self.stencil_desc_pool, None);
			for &framebuffer in &self.framebuffers {
				self.gfx.device.destroy_framebuffer(framebuffer, None);
			}
			self.gfx.device.destroy_pipeline(self.pipeline, None);
			for &image_view in &self.image_views {
				self.gfx.device.destroy_image_view(image_view, None);
			}
			self.gfx.khr_swapchain.destroy_swapchain(self.swapchain, None);
			self.gfx.device.destroy_render_pass(self.render_pass, None);
			self.gfx.khr_surface.destroy_surface(self.surface, None);
		}
	}
}

struct FrameData {
	image_available: vk::Semaphore,
	render_finished: vk::Semaphore,
	frame_finished: vk::Fence,
	cmdpool: vk::CommandPool,
	primary: vk::CommandBuffer,
	secondaries: Vec<vk::CommandBuffer>,
}
impl FrameData {
	fn new(gfx: &Arc<Gfx>) -> Self {
		unsafe {
			let image_available = gfx.device.create_semaphore(&vk::SemaphoreCreateInfo::builder(), None).unwrap();
			let render_finished = gfx.device.create_semaphore(&vk::SemaphoreCreateInfo::builder(), None).unwrap();

			let ci = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
			let frame_finished = gfx.device.create_fence(&ci, None).unwrap();

			let ci = vk::CommandPoolCreateInfo::builder()
				.flags(vk::CommandPoolCreateFlags::TRANSIENT)
				.queue_family_index(gfx.queue_family);
			let cmdpool = gfx.device.create_command_pool(&ci, None).unwrap();

			let ci = vk::CommandBufferAllocateInfo::builder()
				.command_pool(cmdpool)
				.level(vk::CommandBufferLevel::PRIMARY)
				.command_buffer_count(1);
			let primary = gfx.device.allocate_command_buffers(&ci).unwrap()[0];

			Self { image_available, render_finished, frame_finished, cmdpool, primary, secondaries: vec![] }
		}
	}

	fn dispose(&self, device: &Device) {
		unsafe {
			device.free_command_buffers(self.cmdpool, &[self.primary]);
			device.destroy_command_pool(self.cmdpool, None);
			device.destroy_fence(self.frame_finished, None);
			device.destroy_semaphore(self.render_finished, None);
			device.destroy_semaphore(self.image_available, None);
		}
	}
}

fn get_caps(
	gfx: &Gfx,
	surface: vk::SurfaceKHR,
	window: &winit::window::Window,
) -> (vk::SurfaceCapabilitiesKHR, vk::Extent2D) {
	let caps =
		unsafe { gfx.khr_surface.get_physical_device_surface_capabilities(gfx.physical_device, surface) }.unwrap();
	let image_extent = if caps.current_extent.width != u32::MAX {
		caps.current_extent
	} else {
		let (width, height) = window.inner_size().to_physical(1.0).into();
		vk::Extent2D {
			width: max(caps.min_image_extent.width, min(caps.max_image_extent.width, width)),
			height: max(caps.min_image_extent.height, min(caps.max_image_extent.height, height)),
		}
	};

	(caps, image_extent)
}

fn create_swapchain(
	gfx: &Gfx,
	surface: vk::SurfaceKHR,
	caps: &vk::SurfaceCapabilitiesKHR,
	surface_format: &vk::SurfaceFormatKHR,
	image_extent: vk::Extent2D,
	old_swapchain: vk::SwapchainKHR,
) -> (vk::SwapchainKHR, std::vec::Vec<vk::ImageView>) {
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
		.clipped(true)
		.old_swapchain(old_swapchain);
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

	(swapchain, image_views)
}

fn create_pipeline(gfx: &Gfx, image_extent: vk::Extent2D, render_pass: vk::RenderPass) -> vk::Pipeline {
	let name = CStr::from_bytes_with_nul(b"main\0").unwrap();
	let stages = [
		vk::PipelineShaderStageCreateInfo::builder()
			.stage(vk::ShaderStageFlags::VERTEX)
			.module(gfx.vshader)
			.name(name)
			.build(),
		vk::PipelineShaderStageCreateInfo::builder()
			.stage(vk::ShaderStageFlags::FRAGMENT)
			.module(gfx.fshader)
			.name(name)
			.build(),
	];
	let vertex_binding_descriptions = [TriangleVertex::binding_desc()];
	let vertex_attribute_descriptions = TriangleVertex::attribute_descs();
	let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
		.vertex_binding_descriptions(&vertex_binding_descriptions)
		.vertex_attribute_descriptions(&vertex_attribute_descriptions);
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
		.layout(gfx.pipeline_layout)
		.render_pass(render_pass)
		.build()];
	unsafe { gfx.device.create_graphics_pipelines(vk::PipelineCache::null(), &cis, None) }.unwrap()[0]
}

fn create_framebuffers(
	gfx: &Gfx,
	image_views: &[vk::ImageView],
	render_pass: vk::RenderPass,
	image_extent: vk::Extent2D,
) -> Vec<vk::Framebuffer> {
	image_views
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
		.collect()
}

fn create_stencil_desc_pool(gfx: &Gfx, max_sets: u32) -> (vk::DescriptorPool, Vec<vk::DescriptorSet>) {
	let pool_sizes =
		[vk::DescriptorPoolSize::builder().ty(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(max_sets).build()];
	let ci = vk::DescriptorPoolCreateInfo::builder().max_sets(max_sets).pool_sizes(&pool_sizes);
	let desc_pool = unsafe { gfx.device.create_descriptor_pool(&ci, None) }.unwrap();

	let set_layouts = vec![gfx.stencil_desc_layout; max_sets as _];
	let ci = vk::DescriptorSetAllocateInfo::builder().descriptor_pool(desc_pool).set_layouts(&set_layouts);
	let desc_sets = unsafe { gfx.device.allocate_descriptor_sets(&ci) }.unwrap();

	(desc_pool, desc_sets)
}
