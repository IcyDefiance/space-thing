use crate::{physical_device::PhysicalDevice, surface::Surface, Vulkan};
use ash::{
	extensions::{ext, khr},
	version::{EntryV1_0, InstanceV1_0},
	vk, vk_make_version, Instance as VkInstance,
};
use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use std::{
	collections::HashSet,
	ffi::{c_void, CStr},
	sync::Arc,
};

pub struct Instance {
	_vulkan: Arc<Vulkan>,
	pub vk: VkInstance,
	pub khr_surface: khr::Surface,
	#[cfg(windows)]
	pub khr_win32_surface: khr::Win32Surface,
	#[cfg(unix)]
	pub khr_xlib_surface: khr::XlibSurface,
	#[cfg(unix)]
	pub khr_wayland_surface: khr::WaylandSurface,
	#[cfg(debug_assertions)]
	pub debug_utils: ext::DebugUtils,
	#[cfg(debug_assertions)]
	debug_messenger: vk::DebugUtilsMessengerEXT,
}
impl Instance {
	pub fn new(vulkan: Arc<Vulkan>, application_name: &CStr, application_version: Version) -> Arc<Self> {
		let app_info = vk::ApplicationInfo::builder()
			.application_name(&application_name)
			.application_version(application_version.vk);

		let mut exts = vec![b"VK_KHR_surface\0".as_ptr() as _];
		#[cfg(windows)]
		exts.push(b"VK_KHR_win32_surface\0".as_ptr() as _);
		#[cfg(unix)]
		exts.push(b"VK_KHR_xlib_surface\0".as_ptr() as _);
		#[cfg(debug_assertions)]
		exts.push(b"VK_EXT_debug_utils\0".as_ptr() as _);

		#[allow(unused_mut)]
		let mut layers_pref = HashSet::new();
		#[cfg(debug_assertions)]
		layers_pref.insert(CStr::from_bytes_with_nul(b"VK_LAYER_LUNARG_standard_validation\0").unwrap());
		#[cfg(debug_assertions)]
		layers_pref.insert(CStr::from_bytes_with_nul(b"VK_LAYER_LUNARG_monitor\0").unwrap());
		let layers = vulkan.vk.enumerate_instance_layer_properties().unwrap();
		let layers = layers
			.iter()
			.map(|props| unsafe { CStr::from_ptr(props.layer_name.as_ptr()) })
			.collect::<HashSet<_>>()
			.intersection(&layers_pref)
			.map(|ext| ext.as_ptr())
			.collect::<Vec<_>>();

		let ci = vk::InstanceCreateInfo::builder()
			.application_info(&app_info)
			.enabled_layer_names(&layers)
			.enabled_extension_names(&exts);
		let vk = unsafe { vulkan.vk.create_instance(&ci, None) }.unwrap();
		let khr_surface = khr::Surface::new(&vulkan.vk, &vk);
		#[cfg(windows)]
		let khr_win32_surface = khr::Win32Surface::new(&vulkan.vk, &vk);
		#[cfg(unix)]
		let khr_xlib_surface = khr::XlibSurface::new(&vulkan.vk, &vk);
		#[cfg(unix)]
		let khr_wayland_surface = khr::WaylandSurface::new(&vulkan.vk, &vk);
		#[cfg(debug_assertions)]
		let debug_utils = ext::DebugUtils::new(&vulkan.vk, &vk);

		#[cfg(debug_assertions)]
		let debug_messenger = {
			let ci = vk::DebugUtilsMessengerCreateInfoEXT::builder()
				.message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
				.message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
				.pfn_user_callback(Some(user_callback));
			unsafe { debug_utils.create_debug_utils_messenger(&ci, None) }.unwrap()
		};

		Arc::new(Self {
			_vulkan: vulkan,
			vk,
			khr_surface,
			#[cfg(windows)]
			khr_win32_surface,
			#[cfg(unix)]
			khr_xlib_surface,
			#[cfg(unix)]
			khr_wayland_surface,
			#[cfg(debug_assertions)]
			debug_utils,
			#[cfg(debug_assertions)]
			debug_messenger,
		})
	}

	pub fn create_surface<T: HasRawWindowHandle>(self: &Arc<Self>, window: T) -> Arc<Surface<T>> {
		let vk = match window.raw_window_handle() {
			#[cfg(windows)]
			RawWindowHandle::Windows(handle) => {
				let ci = vk::Win32SurfaceCreateInfoKHR::builder().hinstance(handle.hinstance).hwnd(handle.hwnd);
				unsafe { self.khr_win32_surface.create_win32_surface(&ci, None) }.unwrap()
			},
			#[cfg(unix)]
			RawWindowHandle::Xlib(handle) => {
				let ci = vk::XlibSurfaceCreateInfoKHR::builder().dpy(handle.display as _).window(handle.window);
				unsafe { self.khr_xlib_surface.create_xlib_surface(&ci, None) }.unwrap()
			},
			#[cfg(unix)]
			RawWindowHandle::Wayland(handle) => {
				let ci = vk::WaylandSurfaceCreateInfoKHR::builder().display(handle.display).surface(handle.surface);
				unsafe { self.khr_wayland_surface.create_wayland_surface(&ci, None) }.unwrap()
			},
			_ => unimplemented!(),
		};

		unsafe { Surface::from_vk(self.clone(), window, vk) }
	}

	pub fn enumerate_physical_devices<'a>(self: &'a Arc<Instance>) -> impl Iterator<Item = PhysicalDevice<'a>> {
		unsafe { self.vk.enumerate_physical_devices() }
			.unwrap()
			.into_iter()
			.map(move |vk| PhysicalDevice::from_vk(self, vk))
	}
}
impl Drop for Instance {
	fn drop(&mut self) {
		unsafe {
			#[cfg(debug_assertions)]
			self.debug_utils.destroy_debug_utils_messenger(self.debug_messenger, None);
			self.vk.destroy_instance(None);
		}
	}
}

pub struct Version {
	vk: u32,
}
impl Version {
	pub fn new(major: u16, minor: u16, patch: u16) -> Self {
		Self { vk: vk_make_version!(major, minor, patch) }
	}
}

#[cfg(debug_assertions)]
unsafe extern "system" fn user_callback(
	message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
	message_types: vk::DebugUtilsMessageTypeFlagsEXT,
	p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
	_p_user_data: *mut c_void,
) -> vk::Bool32 {
	let callback_data = &*p_callback_data;
	if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE) {
		log::debug!("{:?}: {:?}", message_types, CStr::from_ptr(callback_data.p_message));
	} else if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::INFO) {
		log::info!("{:?}: {:?}", message_types, CStr::from_ptr(callback_data.p_message));
	} else if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
		log::warn!("{:?}: {:?}", message_types, CStr::from_ptr(callback_data.p_message));
	} else {
		log::error!("{:?}: {:?}", message_types, CStr::from_ptr(callback_data.p_message));
	}

	vk::FALSE
}
