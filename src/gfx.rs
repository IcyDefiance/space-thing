pub mod window;

use ash::{
	extensions::ext,
	version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
	vk, vk_make_version, Device, Entry, Instance,
};
use maplit::hashset;
use std::{
	collections::HashSet,
	ffi::{c_void, CStr, CString},
	sync::Arc,
};

pub struct Gfx {
	entry: Entry,
	instance: Instance,
	debug_utils: ext::DebugUtils,
	debug_messenger: vk::DebugUtilsMessengerEXT,
	device: Device,
	queue: vk::Queue,
}
impl Gfx {
	pub fn new() -> Arc<Self> {
		let entry = Entry::new().unwrap();

		let name = CString::new(env!("CARGO_PKG_NAME")).unwrap();
		let app_info = vk::ApplicationInfo::builder().application_name(&name).application_version(vk_make_version!(
			env!("CARGO_PKG_VERSION_MAJOR").parse::<u32>().unwrap(),
			env!("CARGO_PKG_VERSION_MINOR").parse::<u32>().unwrap(),
			env!("CARGO_PKG_VERSION_PATCH").parse::<u32>().unwrap()
		));

		let exts = [b"VK_EXT_debug_utils\0".as_ptr() as _];
		let layers_pref = hashset! { CStr::from_bytes_with_nul(b"VK_LAYER_LUNARG_standard_validation\0").unwrap() };
		let layers = entry.enumerate_instance_extension_properties().unwrap();
		let layers = layers
			.iter()
			.map(|props| unsafe { CStr::from_ptr(props.extension_name.as_ptr()) })
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
		let ci = vk::DebugUtilsMessengerCreateInfoEXT::builder()
			.message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
			.message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
			.pfn_user_callback(Some(user_callback));
		let debug_messenger = unsafe { debug_utils.create_debug_utils_messenger(&ci, None) }.unwrap();

		let pdevice = unsafe { instance.enumerate_physical_devices() }.unwrap()[0];

		let qfam = unsafe { instance.get_physical_device_queue_family_properties(pdevice) }
			.into_iter()
			.enumerate()
			.filter(|(_, props)| props.queue_flags.contains(vk::QueueFlags::GRAPHICS))
			.next()
			.unwrap()
			.0 as u32;
		let qci = [vk::DeviceQueueCreateInfo::builder().queue_family_index(qfam).queue_priorities(&[1.0]).build()];
		let ci = vk::DeviceCreateInfo::builder().queue_create_infos(&qci);
		let device = unsafe { instance.create_device(pdevice, &ci, None) }.unwrap();
		let queue = unsafe { device.get_device_queue(qfam, 0) };

		Arc::new(Self { entry, instance, debug_utils, debug_messenger, device, queue })
	}
}
impl Drop for Gfx {
	fn drop(&mut self) {
		unsafe {
			self.device.destroy_device(None);
			self.debug_utils.destroy_debug_utils_messenger(self.debug_messenger, None);
			self.instance.destroy_instance(None);
		}
	}
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
	}

	vk::FALSE
}
