use crate::{
	device::{Device, Queue},
	instance::Instance,
	surface::Surface,
};
use ash::{version::InstanceV1_0, vk};
use std::sync::Arc;

#[derive(Clone, Copy)]
pub struct PhysicalDevice<'a> {
	instance: &'a Arc<Instance>,
	pub vk: vk::PhysicalDevice,
}
impl<'a> PhysicalDevice<'a> {
	pub fn create_device(
		&self,
		qfams: impl IntoIterator<Item = (QueueFamily<'a>, &'a [f32])>,
	) -> (Arc<Device>, impl Iterator<Item = Arc<Queue>>) {
		let qcis: Vec<_> = qfams
			.into_iter()
			.inspect(|(qfam, _)| assert!(&qfam.physical_device() == self))
			.map(|(qfam, priorities)| {
				vk::DeviceQueueCreateInfo::builder().queue_family_index(qfam.idx).queue_priorities(priorities).build()
			})
			.collect();

		let exts = [b"VK_KHR_swapchain\0".as_ptr() as _];

		let ci = vk::DeviceCreateInfo::builder().queue_create_infos(&qcis).enabled_extension_names(&exts);
		let vk = unsafe { self.instance.vk.create_device(self.vk, &ci, None) }.unwrap();
		let device = Device::from_vk(self.instance.clone(), self.vk, vk);

		let device2 = device.clone();
		let queues = qcis
			.into_iter()
			.map(move |qci| {
				let device = device2.clone();
				(0..qci.queue_count).map(move |idx| unsafe { device.get_queue(qci.queue_family_index, idx) })
			})
			.flatten();

		(device, queues)
	}

	pub fn get_queue_family_properties(self) -> impl Iterator<Item = QueueFamilyProperties<'a>> {
		unsafe { self.instance.vk.get_physical_device_queue_family_properties(self.vk) }
			.into_iter()
			.enumerate()
			.map(move |(i, vk)| QueueFamilyProperties { family: QueueFamily { pdev: self, idx: i as _ }, vk })
	}

	pub fn get_surface_support<T>(&self, qfam: QueueFamily, surface: &Surface<T>) -> bool {
		unsafe { self.instance.khr_surface.get_physical_device_surface_support(self.vk, qfam.idx, surface.vk) }
	}

	pub fn instance(&self) -> &Arc<Instance> {
		self.instance
	}

	pub(crate) fn from_vk(instance: &'a Arc<Instance>, vk: vk::PhysicalDevice) -> Self {
		Self { instance, vk }
	}
}
impl<'a> PartialEq for PhysicalDevice<'a> {
	fn eq(&self, other: &PhysicalDevice) -> bool {
		self.vk == other.vk
	}
}
impl<'a> Eq for PhysicalDevice<'a> {}

pub struct QueueFamilyProperties<'a> {
	family: QueueFamily<'a>,
	vk: vk::QueueFamilyProperties,
}
impl<'a> QueueFamilyProperties<'a> {
	pub fn family(self) -> QueueFamily<'a> {
		self.family
	}

	pub fn queue_flags(&self) -> QueueFlags {
		QueueFlags { vk: self.vk.queue_flags }
	}
}

#[derive(Clone, Copy)]
pub struct QueueFamily<'a> {
	pdev: PhysicalDevice<'a>,
	pub idx: u32,
}
impl<'a> QueueFamily<'a> {
	pub fn physical_device(&self) -> PhysicalDevice {
		self.pdev
	}

	pub(crate) fn from_vk(pdev: PhysicalDevice<'a>, idx: u32) -> Self {
		Self { pdev, idx }
	}
}

#[derive(Clone, Copy)]
pub struct QueueFlags {
	vk: vk::QueueFlags,
}
impl QueueFlags {
	pub fn graphics(self) -> bool {
		self.vk.contains(vk::QueueFlags::GRAPHICS)
	}
}
