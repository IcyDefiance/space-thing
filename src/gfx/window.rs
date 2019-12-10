use crate::gfx::Gfx;
use ash::{version::DeviceV1_0, vk};
use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use std::{
	cmp::{max, min},
	sync::Arc,
	u32,
};
use winit::{EventsLoop, WindowBuilder};

pub struct Window {
	gfx: Arc<Gfx>,
	window: winit::Window,
	surface: vk::SurfaceKHR,
	swapchain: vk::SwapchainKHR,
	image_views: Vec<vk::ImageView>,
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
				println!("{:?}", handle);
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

		Self { gfx: gfx.clone(), window, surface, swapchain, image_views }
	}
}
impl Drop for Window {
	fn drop(&mut self) {
		unsafe {
			for &image_view in &self.image_views {
				self.gfx.device.destroy_image_view(image_view, None);
			}
			self.gfx.khr_swapchain.destroy_swapchain(self.swapchain, None);
			self.gfx.khr_surface.destroy_surface(self.surface, None);
		}
	}
}
