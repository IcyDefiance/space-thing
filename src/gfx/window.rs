use crate::gfx::{Gfx, TriangleVertex};
use ash::{version::DeviceV1_0, vk, Device};
use std::{
	cmp::{max, min},
	iter::empty,
	slice,
	sync::Arc,
	u32,
};
use vulkan::{
	command::{CommandBuffer, CommandPool},
	image::{Format, Framebuffer, ImageView},
	ordered_passes_renderpass,
	pipeline::Pipeline,
	render_pass::RenderPass,
	surface::{ColorSpace, PresentMode, Surface, SurfaceCapabilities},
	swapchain::{CompositeAlphaFlags, Swapchain},
	Extent2D,
};
use winit::{
	event_loop::EventLoop,
	window::{Window as IWindow, WindowBuilder},
};

pub struct Window {
	pub(super) gfx: Arc<Gfx>,
	surface: Arc<Surface<IWindow>>,
	surface_format: vk::SurfaceFormatKHR,
	pub(super) render_pass: Arc<RenderPass>,
	frame_data: [FrameData; 2],
	image_extent: Extent2D,
	present_mode: PresentMode,
	swapchain: Arc<Swapchain<IWindow>>,
	pub(super) pipeline: Pipeline,
	pub(super) framebuffers: Vec<Arc<Framebuffer>>,
	frame: bool,
	recreate_swapchain: bool,
}
impl Window {
	pub fn new(gfx: Arc<Gfx>, event_loop: &EventLoop<()>) -> Self {
		let window = WindowBuilder::new().with_inner_size((1440, 810).into()).build(&event_loop).unwrap();
		let surface = gfx.instance.create_surface(window);
		assert!(gfx.device.physical_device().get_surface_support(gfx.queue.family(), &surface));

		let surface_format = gfx
			.device
			.physical_device()
			.get_surface_formats(&surface)
			.into_iter()
			.max_by_key(|format| {
				format.format == Format::B8G8R8A8_UNORM && format.color_space == ColorSpace::SRGB_NONLINEAR
			})
			.unwrap();

		let render_pass = ordered_passes_renderpass!(gfx.device.clone(),
			attachments: { color: { load: Clear, store: Store, format: surface_format.format, samples: 1, } },
			passes: [{ color: [color], depth_stencil: {}, input: [] }]
		);

		let (caps, image_extent) = get_caps(&gfx, &surface);
		let present_mode = gfx
			.device
			.physical_device()
			.get_surface_present_modes(&surface)
			.into_iter()
			.min_by_key(|&mode| match mode {
				PresentMode::MAILBOX => 0,
				PresentMode::IMMEDIATE => 1,
				PresentMode::FIFO_RELAXED => 2,
				PresentMode::FIFO => 3,
				_ => 4,
			})
			.unwrap();

		let (swapchain, image_views) =
			create_swapchain(&gfx, surface.clone(), &caps, &surface_format, image_extent, present_mode, None);
		let pipeline = create_pipeline(&gfx, image_extent, render_pass.clone());
		let framebuffers = create_framebuffers(&render_pass, image_views, image_extent);

		let frame_data = [FrameData::new(&gfx), FrameData::new(&gfx)];

		Self {
			gfx,
			surface,
			surface_format,
			render_pass,
			frame_data,
			image_extent,
			present_mode,
			swapchain,
			pipeline,
			framebuffers,
			frame: false,
			recreate_swapchain: false,
		}
	}

	pub fn draw(&mut self) {
		unsafe {
			if self.recreate_swapchain {
				self.recreate_swapchain();
			}

			let frame = self.frame as usize;
			let frame_data = &mut self.frame_data[frame];

			let res = self.gfx.device.khr_swapchain.acquire_next_image(
				self.swapchain.vk,
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

			self.gfx.device.vk.wait_for_fences(&[frame_data.frame_finished], false, !0).unwrap();
			self.gfx.device.vk.reset_fences(&[frame_data.frame_finished]).unwrap();
			self.frame = !self.frame;

			let framebuffer = &self.framebuffers[image_uidx];

			frame_data.cmdpool.reset(false);

			// TODO: replace with real volumes
			let volumes_len = 2;
			if frame_data.secondaries.len() < volumes_len {
				let newcmds = frame_data
					.cmdpool
					.allocate_command_buffers(true, (volumes_len - frame_data.secondaries.len()) as _);
				frame_data.secondaries.extend(newcmds);
			}
			for (_, cmd) in (0..volumes_len).zip(&frame_data.secondaries) {
				let flags =
					vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT | vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE;
				let inherit = vk::CommandBufferInheritanceInfo::builder()
					.render_pass(self.render_pass.vk)
					.subpass(0)
					.framebuffer(framebuffer.vk);
				let info = vk::CommandBufferBeginInfo::builder().flags(flags).inheritance_info(&inherit);
				let cmd = cmd.inner.read().unwrap().vk;
				self.gfx.device.vk.begin_command_buffer(cmd, &info).unwrap();
				self.gfx.device.vk.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline.vk);
				self.gfx.device.vk.cmd_bind_vertex_buffers(cmd, 0, &[self.gfx.triangle.vk], &[0]);
				self.gfx.device.vk.cmd_draw(cmd, 3, 1, 0, 0);
				self.gfx.device.vk.end_command_buffer(cmd).unwrap();
			}

			let bi = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
			let cmd = frame_data.primary.inner.read().unwrap().vk;
			self.gfx.device.vk.begin_command_buffer(cmd, &bi).unwrap();
			let ci = vk::RenderPassBeginInfo::builder()
				.render_pass(self.render_pass.vk)
				.framebuffer(framebuffer.vk)
				.render_area(vk::Rect2D::builder().extent(self.image_extent).build())
				.clear_values(&[vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } }]);
			self.gfx.device.vk.cmd_begin_render_pass(cmd, &ci, vk::SubpassContents::SECONDARY_COMMAND_BUFFERS);
			let secondaries: Vec<_> =
				frame_data.secondaries[0..volumes_len].iter().map(|x| x.inner.read().unwrap().vk).collect();
			self.gfx.device.vk.cmd_execute_commands(cmd, &secondaries);
			self.gfx.device.vk.cmd_end_render_pass(cmd);
			self.gfx.device.vk.end_command_buffer(cmd).unwrap();
			let submits = [vk::SubmitInfo::builder()
				.wait_semaphores(&[frame_data.image_available])
				.wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
				.command_buffers(&[cmd])
				.signal_semaphores(&[frame_data.render_finished])
				.build()];
			self.gfx.device.vk.queue_submit(self.gfx.queue.vk, &submits, frame_data.frame_finished).unwrap();

			let ci = vk::PresentInfoKHR::builder()
				.wait_semaphores(slice::from_ref(&frame_data.render_finished))
				.swapchains(slice::from_ref(&self.swapchain.vk))
				.image_indices(slice::from_ref(&image_idx));
			match self.gfx.device.khr_swapchain.queue_present(self.gfx.queue.vk, &ci) {
				Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.recreate_swapchain = true,
				Ok(false) => (),
				Err(err) => panic!(err),
			}
		}
	}

	fn recreate_swapchain(&mut self) {
		unsafe {
			self.gfx
				.device
				.vk
				.wait_for_fences(&[self.frame_data[(!self.frame) as usize].frame_finished], false, !0)
				.unwrap();

			let (caps, image_extent) = get_caps(&self.gfx, &self.surface);
			let (swapchain, image_views) = create_swapchain(
				&self.gfx,
				self.surface.clone(),
				&caps,
				&self.surface_format,
				image_extent,
				self.present_mode,
				Some(&self.swapchain),
			);
			self.swapchain = swapchain;

			self.pipeline = create_pipeline(&self.gfx, image_extent, self.render_pass.clone());
			self.framebuffers = create_framebuffers(&self.render_pass, image_views, image_extent);

			self.image_extent = image_extent;
		}

		self.recreate_swapchain = false;
	}
}
impl Drop for Window {
	fn drop(&mut self) {
		unsafe {
			self.gfx
				.device
				.vk
				.wait_for_fences(&[self.frame_data[(!self.frame) as usize].frame_finished], false, !0)
				.unwrap();

			self.frame_data[0].dispose(&self.gfx.device.vk);
			self.frame_data[1].dispose(&self.gfx.device.vk);
		}
	}
}

struct FrameData {
	image_available: vk::Semaphore,
	render_finished: vk::Semaphore,
	frame_finished: vk::Fence,
	cmdpool: Arc<CommandPool>,
	primary: Arc<CommandBuffer>,
	secondaries: Vec<Arc<CommandBuffer>>,
}
impl FrameData {
	fn new(gfx: &Arc<Gfx>) -> Self {
		unsafe {
			let image_available = gfx.device.vk.create_semaphore(&vk::SemaphoreCreateInfo::builder(), None).unwrap();
			let render_finished = gfx.device.vk.create_semaphore(&vk::SemaphoreCreateInfo::builder(), None).unwrap();
			let ci = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
			let frame_finished = gfx.device.vk.create_fence(&ci, None).unwrap();
			let cmdpool = gfx.device.create_command_pool(gfx.queue.family(), true);
			let primary = cmdpool.allocate_command_buffers(false, 1).next().unwrap();
			Self { image_available, render_finished, frame_finished, cmdpool, primary, secondaries: vec![] }
		}
	}

	fn dispose(&self, device: &Device) {
		unsafe {
			device.destroy_fence(self.frame_finished, None);
			device.destroy_semaphore(self.render_finished, None);
			device.destroy_semaphore(self.image_available, None);
		}
	}
}

fn get_caps(gfx: &Gfx, surface: &Surface<IWindow>) -> (SurfaceCapabilities, Extent2D) {
	let caps = gfx.device.physical_device().get_surface_capabilities(surface);
	let image_extent = if caps.current_extent.width != u32::MAX {
		caps.current_extent
	} else {
		let (width, height) = surface.window().inner_size().to_physical(1.0).into();
		Extent2D {
			width: max(caps.min_image_extent.width, min(caps.max_image_extent.width, width)),
			height: max(caps.min_image_extent.height, min(caps.max_image_extent.height, height)),
		}
	};

	(caps, image_extent)
}

fn create_swapchain<T: 'static>(
	gfx: &Gfx,
	surface: Arc<Surface<T>>,
	caps: &SurfaceCapabilities,
	surface_format: &vk::SurfaceFormatKHR,
	image_extent: Extent2D,
	present_mode: PresentMode,
	old_swapchain: Option<&Swapchain<T>>,
) -> (Arc<Swapchain<T>>, Vec<Arc<ImageView>>) {
	let (swapchain, images) = gfx.device.create_swapchain(
		surface,
		caps.min_image_count + 1,
		surface_format.format,
		surface_format.color_space,
		image_extent,
		empty(),
		caps.current_transform,
		CompositeAlphaFlags::OPAQUE,
		present_mode,
		old_swapchain,
	);

	let image_views = images
		.map(|image| {
			let range = vk::ImageSubresourceRange::builder()
				.aspect_mask(vk::ImageAspectFlags::COLOR)
				.level_count(1)
				.layer_count(1)
				.build();
			gfx.device.create_image_view(image, surface_format.format, range)
		})
		.collect();

	(swapchain, image_views)
}

fn create_pipeline(gfx: &Gfx, image_extent: Extent2D, render_pass: Arc<RenderPass>) -> Pipeline {
	gfx.device
		.build_pipeline(gfx.layout.clone(), render_pass)
		.vertex_shader(gfx.vshader.clone())
		.fragment_shader(gfx.fshader.clone())
		.vertex_input::<TriangleVertex>()
		.viewports(&[vk::Viewport::builder()
			.width(image_extent.width as _)
			.height(image_extent.height as _)
			.max_depth(1.0)
			.build()])
		.build()
}

fn create_framebuffers(
	render_pass: &Arc<RenderPass>,
	image_views: Vec<Arc<ImageView>>,
	image_extent: Extent2D,
) -> Vec<Arc<Framebuffer>> {
	image_views
		.into_iter()
		.map(|view| {
			render_pass.device().create_framebuffer(
				render_pass.clone(),
				vec![view],
				image_extent.width,
				image_extent.height,
			)
		})
		.collect()
}
