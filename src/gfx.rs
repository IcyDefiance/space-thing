pub mod buffer;
pub mod volume;
pub mod window;

use crate::fs::read_bytes;
use ash::{
	extensions::{ext, khr},
	version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
	vk, vk_make_version, Device, Entry, Instance,
};
use maplit::hashset;
use std::{
	collections::HashSet,
	ffi::{c_void, CStr, CString},
	slice,
	sync::Arc,
};
use vk_mem::{Allocator, AllocatorCreateInfo};

pub struct Gfx {
	_entry: Entry,
	instance: Instance,
	debug_utils: ext::DebugUtils,
	khr_surface: khr::Surface,
	#[cfg(windows)]
	khr_win32_surface: khr::Win32Surface,
	#[cfg(unix)]
	khr_xlib_surface: khr::XlibSurface,
	#[cfg(unix)]
	khr_wayland_surface: khr::WaylandSurface,
	debug_messenger: vk::DebugUtilsMessengerEXT,
	physical_device: vk::PhysicalDevice,
	queue_family: u32,
	device: Device,
	khr_swapchain: khr::Swapchain,
	queue: vk::Queue,
	cmdpool: vk::CommandPool,
	cmdpool_transient: vk::CommandPool,
	layout: vk::PipelineLayout,
	allocator: Allocator,
	vshader: vk::ShaderModule,
	fshader: vk::ShaderModule,
}
impl Gfx {
	pub async fn new() -> Arc<Self> {
		// start reading files now to use later
		let vert_spv = read_bytes("build/shader.vert.spv");
		let frag_spv = read_bytes("build/shader.frag.spv");

		let entry = Entry::new().unwrap();

		let name = CString::new(env!("CARGO_PKG_NAME")).unwrap();
		let app_info = vk::ApplicationInfo::builder().application_name(&name).application_version(vk_make_version!(
			env!("CARGO_PKG_VERSION_MAJOR").parse::<u32>().unwrap(),
			env!("CARGO_PKG_VERSION_MINOR").parse::<u32>().unwrap(),
			env!("CARGO_PKG_VERSION_PATCH").parse::<u32>().unwrap()
		));
		let mut exts = vec![b"VK_EXT_debug_utils\0".as_ptr() as _, b"VK_KHR_surface\0".as_ptr() as _];
		if cfg!(windows) {
			exts.push(b"VK_KHR_win32_surface\0".as_ptr() as _);
		} else {
			exts.push(b"VK_KHR_xlib_surface\0".as_ptr() as _);
		}
		let layers_pref = hashset! {
			CStr::from_bytes_with_nul(b"VK_LAYER_LUNARG_standard_validation\0").unwrap(),
			CStr::from_bytes_with_nul(b"VK_LAYER_LUNARG_monitor\0").unwrap(),
		};
		let layers = entry.enumerate_instance_layer_properties().unwrap();
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
		let instance = unsafe { entry.create_instance(&ci, None) }.unwrap();
		let debug_utils = ext::DebugUtils::new(&entry, &instance);
		let khr_surface = khr::Surface::new(&entry, &instance);
		#[cfg(windows)]
		let khr_win32_surface = khr::Win32Surface::new(&entry, &instance);
		#[cfg(unix)]
		let khr_xlib_surface = khr::XlibSurface::new(&entry, &instance);
		#[cfg(unix)]
		let khr_wayland_surface = khr::WaylandSurface::new(&entry, &instance);

		let ci = vk::DebugUtilsMessengerCreateInfoEXT::builder()
			.message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
			.message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
			.pfn_user_callback(Some(user_callback));
		let debug_messenger = unsafe { debug_utils.create_debug_utils_messenger(&ci, None) }.unwrap();

		let physical_device = unsafe { instance.enumerate_physical_devices() }.unwrap()[0];

		let queue_family = unsafe { instance.get_physical_device_queue_family_properties(physical_device) }
			.into_iter()
			.enumerate()
			.filter(|(_, props)| props.queue_flags.contains(vk::QueueFlags::GRAPHICS))
			.next()
			.unwrap()
			.0 as u32;
		let qci =
			[vk::DeviceQueueCreateInfo::builder().queue_family_index(queue_family).queue_priorities(&[1.0]).build()];

		let exts = [b"VK_KHR_swapchain\0".as_ptr() as _];
		let ci = vk::DeviceCreateInfo::builder().queue_create_infos(&qci).enabled_extension_names(&exts);
		let device = unsafe { instance.create_device(physical_device, &ci, None) }.unwrap();
		let khr_swapchain = khr::Swapchain::new(&instance, &device);

		let queue = unsafe { device.get_device_queue(queue_family, 0) };

		let ci = vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family);
		let cmdpool = unsafe { device.create_command_pool(&ci, None) }.unwrap();

		let ci = ci.flags(vk::CommandPoolCreateFlags::TRANSIENT);
		let cmdpool_transient = unsafe { device.create_command_pool(&ci, None) }.unwrap();

		let ci = vk::PipelineLayoutCreateInfo::builder();
		let layout = unsafe { device.create_pipeline_layout(&ci, None) }.unwrap();

		let ci = AllocatorCreateInfo {
			physical_device,
			device: device.clone(),
			instance: instance.clone(),
			..AllocatorCreateInfo::default()
		};
		let allocator = Allocator::new(&ci).unwrap();

		let vshader = create_shader(&device, &vert_spv.await.unwrap());
		let fshader = create_shader(&device, &frag_spv.await.unwrap());

		Arc::new(Self {
			_entry: entry,
			instance,
			debug_utils,
			khr_surface,
			#[cfg(windows)]
			khr_win32_surface,
			#[cfg(unix)]
			khr_xlib_surface,
			#[cfg(unix)]
			khr_wayland_surface,
			debug_messenger,
			physical_device,
			queue_family,
			device,
			khr_swapchain,
			queue,
			cmdpool,
			cmdpool_transient,
			layout,
			allocator,
			vshader,
			fshader,
		})
	}
}
impl Drop for Gfx {
	fn drop(&mut self) {
		unsafe {
			self.device.destroy_shader_module(self.fshader, None);
			self.device.destroy_shader_module(self.vshader, None);
			self.allocator.destroy();
			self.device.destroy_pipeline_layout(self.layout, None);
			self.device.destroy_command_pool(self.cmdpool_transient, None);
			self.device.destroy_command_pool(self.cmdpool, None);
			self.device.destroy_device(None);
			self.debug_utils.destroy_debug_utils_messenger(self.debug_messenger, None);
			self.instance.destroy_instance(None);
		}
	}
}

fn create_shader(device: &Device, code: &[u8]) -> vk::ShaderModule {
	let code = unsafe { slice::from_raw_parts(code.as_ptr() as _, code.len() / 4) };
	let ci = vk::ShaderModuleCreateInfo::builder().code(code);
	unsafe { device.create_shader_module(&ci, None) }.unwrap()
}

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
		panic!();
	}

	vk::FALSE
}
