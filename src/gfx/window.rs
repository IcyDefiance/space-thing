use crate::gfx::{volume::Volume, Gfx};
use ash::{version::DeviceV1_0, vk, Device};
use memoffset::offset_of;
use nalgebra::{Vector2, Vector3};
use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use std::{
	cmp::{max, min},
	ffi::CStr,
	mem::size_of,
	slice,
	sync::{Arc, Mutex},
	u32,
};
use winit::{EventsLoop, WindowBuilder};

pub struct Window {
	window: winit::Window,
	surface: vk::SurfaceKHR,
	surface_format: vk::SurfaceFormatKHR,
	pub(super) render_pass: vk::RenderPass,
	frame_data: [FrameData; 2],
	pub(super) inner: Mutex<WindowInner>,
}
impl Window {
	pub fn new(gfx: Arc<Gfx>, events_loop: &EventsLoop) -> Self {
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

		let frame_data = [FrameData::new(&gfx), FrameData::new(&gfx)];

		let inner = Mutex::new(WindowInner {
			gfx,
			image_extent,
			swapchain,
			image_views,
			pipeline,
			framebuffers,
			frame: false,
			volumes: [vec![], vec![]],
			recreate_swapchain: false,
		});

		Self { window, surface, surface_format, render_pass, frame_data, inner }
	}

	pub fn draw(&mut self, volumes: Vec<Arc<Volume>>) {
		unsafe {
			let mut inner = self.inner.lock().unwrap();

			if inner.recreate_swapchain {
				self.recreate_swapchain(&mut *inner);
			}

			let frame = inner.frame as usize;
			let frame_data = &mut self.frame_data[frame];

			let res = inner.gfx.khr_swapchain.acquire_next_image(
				inner.swapchain,
				!0,
				frame_data.image_available,
				vk::Fence::null(),
			);
			let image_idx = match res {
				Ok((idx, suboptimal)) => {
					if suboptimal {
						inner.recreate_swapchain = true;
					}
					idx
				},
				Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
					inner.recreate_swapchain = true;
					return;
				},
				Err(err) => panic!(err),
			};
			let image_uidx = image_idx as usize;

			inner.gfx.device.wait_for_fences(&[frame_data.frame_finished], false, !0).unwrap();
			inner.gfx.device.reset_fences(&[frame_data.frame_finished]).unwrap();
			inner.frame = !inner.frame;

			let framebuffer = inner.framebuffers[image_uidx];

			inner.gfx.device.reset_command_pool(frame_data.cmdpool, vk::CommandPoolResetFlags::empty()).unwrap();

			if frame_data.secondaries.len() < volumes.len() {
				let ci = vk::CommandBufferAllocateInfo::builder()
					.command_pool(frame_data.cmdpool)
					.level(vk::CommandBufferLevel::SECONDARY)
					.command_buffer_count((volumes.len() - frame_data.secondaries.len()) as _);
				frame_data.secondaries.extend(inner.gfx.device.allocate_command_buffers(&ci).unwrap());
			}
			for (vol, &cmd) in volumes.iter().zip(&frame_data.secondaries) {
				let flags =
					vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT | vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE;
				let inherit = vk::CommandBufferInheritanceInfo::builder()
					.render_pass(self.render_pass)
					.subpass(0)
					.framebuffer(framebuffer);
				let info = vk::CommandBufferBeginInfo::builder().flags(flags).inheritance_info(&inherit);
				inner.gfx.device.begin_command_buffer(cmd, &info).unwrap();
				inner.gfx.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, inner.pipeline);
				inner.gfx.device.cmd_bind_vertex_buffers(cmd, 0, &[vol.vertices.buf], &[0]);
				inner.gfx.device.cmd_bind_index_buffer(cmd, vol.indices.buf, 0, vk::IndexType::UINT32);
				inner.gfx.device.cmd_draw_indexed(cmd, vol.indices.len as _, 1, 0, 0, 0);
				inner.gfx.device.end_command_buffer(cmd).unwrap();
			}

			let bi = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
			inner.gfx.device.begin_command_buffer(frame_data.primary, &bi).unwrap();
			let ci = vk::RenderPassBeginInfo::builder()
				.render_pass(self.render_pass)
				.framebuffer(framebuffer)
				.render_area(vk::Rect2D::builder().extent(inner.image_extent).build())
				.clear_values(&[vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } }]);
			inner.gfx.device.cmd_begin_render_pass(
				frame_data.primary,
				&ci,
				vk::SubpassContents::SECONDARY_COMMAND_BUFFERS,
			);
			inner.gfx.device.cmd_execute_commands(frame_data.primary, &frame_data.secondaries[0..volumes.len()]);
			inner.gfx.device.cmd_end_render_pass(frame_data.primary);
			inner.gfx.device.end_command_buffer(frame_data.primary).unwrap();
			let submits = [vk::SubmitInfo::builder()
				.wait_semaphores(&[frame_data.image_available])
				.wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
				.command_buffers(&[frame_data.primary])
				.signal_semaphores(&[frame_data.render_finished])
				.build()];
			inner.gfx.device.queue_submit(inner.gfx.queue, &submits, frame_data.frame_finished).unwrap();

			inner.volumes[frame] = volumes;

			let ci = vk::PresentInfoKHR::builder()
				.wait_semaphores(slice::from_ref(&frame_data.render_finished))
				.swapchains(slice::from_ref(&inner.swapchain))
				.image_indices(slice::from_ref(&image_idx));
			match inner.gfx.khr_swapchain.queue_present(inner.gfx.queue, &ci) {
				Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => inner.recreate_swapchain = true,
				Ok(false) => (),
				Err(err) => panic!(err),
			}
		}
	}

	fn recreate_swapchain(&self, inner: &mut WindowInner) {
		unsafe {
			inner.gfx.device.wait_for_fences(&[self.frame_data[inner.last_frame()].frame_finished], false, !0).unwrap();

			for &framebuffer in &inner.framebuffers {
				inner.gfx.device.destroy_framebuffer(framebuffer, None);
			}
			inner.gfx.device.destroy_pipeline(inner.pipeline, None);
			for &image_view in &inner.image_views {
				inner.gfx.device.destroy_image_view(image_view, None);
			}

			let (caps, image_extent) = get_caps(&inner.gfx, self.surface, &self.window);
			let (swapchain, image_views) =
				create_swapchain(&inner.gfx, self.surface, &caps, &self.surface_format, image_extent, inner.swapchain);
			inner.gfx.khr_swapchain.destroy_swapchain(inner.swapchain, None);
			inner.swapchain = swapchain;
			inner.image_views = image_views;

			inner.pipeline = create_pipeline(&inner.gfx, image_extent, self.render_pass);
			inner.framebuffers = create_framebuffers(&inner.gfx, &inner.image_views, self.render_pass, image_extent);
		}
	}
}
impl Drop for Window {
	fn drop(&mut self) {
		unsafe {
			let inner = self.inner.lock().unwrap();

			inner.gfx.device.wait_for_fences(&[self.frame_data[inner.last_frame()].frame_finished], false, !0).unwrap();

			self.frame_data[0].dispose(&inner.gfx.device);
			self.frame_data[1].dispose(&inner.gfx.device);

			for &framebuffer in &inner.framebuffers {
				inner.gfx.device.destroy_framebuffer(framebuffer, None);
			}
			inner.gfx.device.destroy_pipeline(inner.pipeline, None);
			for &image_view in &inner.image_views {
				inner.gfx.device.destroy_image_view(image_view, None);
			}
			inner.gfx.khr_swapchain.destroy_swapchain(inner.swapchain, None);
			inner.gfx.device.destroy_render_pass(self.render_pass, None);
			inner.gfx.khr_surface.destroy_surface(self.surface, None);
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

pub(super) struct WindowInner {
	pub(super) gfx: Arc<Gfx>,
	image_extent: vk::Extent2D,
	swapchain: vk::SwapchainKHR,
	image_views: Vec<vk::ImageView>,
	pub(super) pipeline: vk::Pipeline,
	pub(super) framebuffers: Vec<vk::Framebuffer>,
	frame: bool,
	volumes: [Vec<Arc<Volume>>; 2],
	recreate_swapchain: bool,
}
impl WindowInner {
	fn last_frame(&self) -> usize {
		(!self.frame) as usize
	}
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Vertex {
	pub pos: Vector2<f32>,
	pub color: Vector3<f32>,
}
impl Vertex {
	fn binding_desc() -> vk::VertexInputBindingDescription {
		vk::VertexInputBindingDescription::builder()
			.binding(0)
			.stride(size_of::<Vertex>() as _)
			.input_rate(vk::VertexInputRate::VERTEX)
			.build()
	}

	fn attribute_descs() -> [vk::VertexInputAttributeDescription; 2] {
		[
			vk::VertexInputAttributeDescription::builder()
				.binding(0)
				.location(0)
				.format(vk::Format::R32G32_SFLOAT)
				.offset(offset_of!(Self, pos) as _)
				.build(),
			vk::VertexInputAttributeDescription::builder()
				.binding(0)
				.location(1)
				.format(vk::Format::R32G32B32_SFLOAT)
				.offset(offset_of!(Self, color) as _)
				.build(),
		]
	}
}

fn get_caps(gfx: &Gfx, surface: vk::SurfaceKHR, window: &winit::Window) -> (vk::SurfaceCapabilitiesKHR, vk::Extent2D) {
	let caps =
		unsafe { gfx.khr_surface.get_physical_device_surface_capabilities(gfx.physical_device, surface) }.unwrap();
	let image_extent = if caps.current_extent.width != u32::MAX {
		caps.current_extent
	} else {
		let (width, height) = window.get_inner_size().unwrap().to_physical(1.0).into();
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
	let vertex_binding_descriptions = [Vertex::binding_desc()];
	let vertex_attribute_descriptions = Vertex::attribute_descs();
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
		.layout(gfx.layout)
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
