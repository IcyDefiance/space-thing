pub use ash::vk::BufferUsageFlags;

use crate::{
	buffer::BufferInit,
	command::{CommandBuffer, CommandPool},
	instance::Instance,
	physical_device::{PhysicalDevice, QueueFamily},
	pipeline::PipelineLayout,
	shader::ShaderModule,
	sync::Fence,
};
use ash::{extensions::khr, version::DeviceV1_0, vk, Device as VkDevice};
use std::{mem::size_of, sync::Arc};
use typenum::Bit;
use vk_mem::{AllocationCreateInfo, Allocator, AllocatorCreateInfo, MemoryUsage};

pub struct Device {
	instance: Arc<Instance>,
	physical_device: vk::PhysicalDevice,
	pub vk: VkDevice,
	pub khr_swapchain: khr::Swapchain,
	pub allocator: Allocator,
}
impl Device {
	pub fn create_buffer_slice<T, CPU: Bit>(
		self: &Arc<Device>,
		len: usize,
		_: CPU,
		usage: BufferUsageFlags,
	) -> BufferInit<[T], CPU> {
		let size = size_of::<T>() as u64 * len as u64;

		let ci = ash::vk::BufferCreateInfo::builder().size(size).usage(usage).build();

		let usage = if CPU::BOOL { MemoryUsage::CpuOnly } else { MemoryUsage::GpuOnly };
		let aci = AllocationCreateInfo { usage, ..Default::default() };

		let (vk, alloc, _) = self.allocator.create_buffer(&ci, &aci).unwrap();

		BufferInit::from_vk(self.clone(), vk, alloc, size)
	}

	pub fn create_command_pool<'a>(self: &Arc<Device>, family: QueueFamily<'a>, transient: bool) -> Arc<CommandPool> {
		unsafe {
			let mut flags = vk::CommandPoolCreateFlags::empty();
			if transient {
				flags |= vk::CommandPoolCreateFlags::TRANSIENT;
			};

			let ci = vk::CommandPoolCreateInfo::builder().flags(flags).queue_family_index(family.idx);
			let vk = self.vk.create_command_pool(&ci, None).unwrap();
			CommandPool::from_vk(self.clone(), family.idx, vk)
		}
	}

	pub(crate) fn create_fence(self: &Arc<Device>, signalled: bool, resources: Vec<Arc<CommandBuffer>>) -> Fence {
		unsafe {
			let mut flags = vk::FenceCreateFlags::empty();
			if signalled {
				flags |= vk::FenceCreateFlags::SIGNALED;
			}

			let vk = self.vk.create_fence(&vk::FenceCreateInfo::builder().flags(flags), None).unwrap();
			Fence::from_vk(self.clone(), vk, resources)
		}
	}

	pub fn create_pipeline_layout(self: &Arc<Device>) -> PipelineLayout {
		let ci = vk::PipelineLayoutCreateInfo::builder();
		let vk = unsafe { self.vk.create_pipeline_layout(&ci, None) }.unwrap();
		PipelineLayout::from_vk(self.clone(), vk)
	}

	pub unsafe fn create_shader_module(self: &Arc<Device>, code: &[u32]) -> ShaderModule {
		let ci = vk::ShaderModuleCreateInfo::builder().code(code);
		let vk = self.vk.create_shader_module(&ci, None).unwrap();
		ShaderModule::from_vk(self.clone(), vk)
	}

	pub fn physical_device(&self) -> PhysicalDevice {
		PhysicalDevice::from_vk(&self.instance, self.physical_device)
	}

	pub(crate) fn from_vk(instance: Arc<Instance>, physical_device: vk::PhysicalDevice, vk: VkDevice) -> Arc<Self> {
		let khr_swapchain = khr::Swapchain::new(&instance.vk, &vk);

		let ci = AllocatorCreateInfo {
			physical_device,
			device: vk.clone(),
			instance: instance.vk.clone(),
			..AllocatorCreateInfo::default()
		};
		let allocator = Allocator::new(&ci).unwrap();

		Arc::new(Self { instance, physical_device, vk, khr_swapchain, allocator })
	}

	pub(crate) unsafe fn get_queue(self: &Arc<Self>, queue_family_index: u32, queue_index: u32) -> Arc<Queue> {
		let vk = self.vk.get_device_queue(queue_family_index, queue_index);

		Arc::new(Queue { device: self.clone(), family: queue_family_index, vk })
	}
}
impl Drop for Device {
	fn drop(&mut self) {
		self.allocator.destroy();
		unsafe { self.vk.destroy_device(None) };
	}
}

pub struct Queue {
	device: Arc<Device>,
	family: u32,
	pub vk: vk::Queue,
}
impl Queue {
	pub fn device(&self) -> &Arc<Device> {
		&self.device
	}

	pub fn family(&self) -> QueueFamily {
		QueueFamily::from_vk(self.device.physical_device(), self.family)
	}

	pub fn submit(self: &Arc<Self>, cmd: Arc<CommandBuffer>) -> SubmitFuture {
		assert!(cmd.pool.queue_family == self.family);

		SubmitFuture { queue: self.clone(), cmd }
	}
}

pub struct SubmitFuture {
	queue: Arc<Queue>,
	cmd: Arc<CommandBuffer>,
}
impl SubmitFuture {
	pub fn end(self) -> Fence {
		let fence = self.queue.device.create_fence(false, vec![self.cmd.clone()]);

		let cmd_inner = self.cmd.inner.read().unwrap();
		let submits = [vk::SubmitInfo::builder().command_buffers(&[cmd_inner.vk]).build()];
		unsafe { self.queue.device().vk.queue_submit(self.queue.vk, &submits, fence.vk) }.unwrap();

		fence
	}
}
