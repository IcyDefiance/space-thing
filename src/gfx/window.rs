use crate::gfx::{Gfx, TriangleVertex};
use ash::{version::DeviceV1_0, vk, Device};
use std::{
	cmp::{max, min},
	iter::{empty, once},
	slice,
	sync::Arc,
	u32,
};
use vulkan::{
	command::{ClearValue, CommandPool, InheritanceInfo},
	image::{Format, Framebuffer, ImageView},
	ordered_passes_renderpass,
	pipeline::Pipeline,
	render_pass::RenderPass,
	surface::{ColorSpace, PresentMode, Surface, SurfaceCapabilities},
	swapchain::{CompositeAlphaFlags, Swapchain},
	sync::Fence,
	Extent2D, Rect2D,
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
	pub(super) pipeline: Arc<Pipeline>,
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
		if self.recreate_swapchain {
			self.recreate_swapchain();
		}

		let res = self.swapchain.acquire_next_image(!0);
		let (image_idx, future) = match res {
			Ok((idx, suboptimal, future)) => {
				if suboptimal {
					self.recreate_swapchain = true;
				}
				(idx, future)
			},
			Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
				self.recreate_swapchain = true;
				return;
			},
			Err(err) => panic!(err),
		};
		let image_uidx = image_idx as usize;

		let frame = self.frame as usize;
		if let Some(fence) = self.frame_data[frame].fence.take() {
			fence.wait();
		}
		self.frame = !self.frame;

		let framebuffer = &self.framebuffers[image_uidx];

		self.frame_data[frame].cmdpool.reset(false);

		// TODO: replace with real volumes
		let secondaries = (0..2).map(|_| {
			let inherit = InheritanceInfo {
				render_pass: self.render_pass.clone(),
				subpass: 0,
				framebuffer: Some(framebuffer.clone()),
			};
			self.frame_data[frame]
				.cmdpool
				.record_secondary(true, false, Some(inherit))
				.bind_pipeline(self.pipeline.clone())
				.bind_vertex_buffers(0, once(self.gfx.triangle.clone() as _), &[0])
				.draw(3, 1, 0, 0)
				.build()
		});

		let primary = self.frame_data[frame]
			.cmdpool
			.record(true, false)
			.begin_render_pass(
				self.render_pass.clone(),
				framebuffer.clone(),
				Rect2D::builder().extent(self.image_extent).build(),
				&[ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } }],
			)
			.execute_commands(secondaries)
			.end_render_pass()
			.build();
		let (fence, future) = self.gfx.queue.submit_after(future, primary).flush();
		self.frame_data[frame].fence = Some(fence);

		match Swapchain::present_after(future, self.gfx.queue.clone(), &[self.swapchain.clone()], &[image_idx]) {
			Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.recreate_swapchain = true,
			Ok(false) => (),
			Err(err) => panic!(err),
		}
	}

	fn recreate_swapchain(&mut self) {
		self.frame_data[(!self.frame) as usize].fence.as_ref().unwrap().wait();

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

		self.recreate_swapchain = false;
	}
}

struct FrameData {
	cmdpool: Arc<CommandPool>,
	fence: Option<Fence>,
}
impl FrameData {
	fn new(gfx: &Arc<Gfx>) -> Self {
		let cmdpool = gfx.device.create_command_pool(gfx.queue.family(), true);
		Self { cmdpool, fence: None }
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

fn create_pipeline(gfx: &Gfx, image_extent: Extent2D, render_pass: Arc<RenderPass>) -> Arc<Pipeline> {
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
