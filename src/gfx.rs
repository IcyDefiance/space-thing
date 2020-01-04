pub mod buffer;
pub mod image;
pub mod window;
pub mod world;

use crate::fs::read_bytes;
use ash::{
	extensions::khr,
	version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
	vk, vk_make_version, Device, Entry, Instance,
};
use buffer::create_device_local_buffer;
use memoffset::offset_of;
use nalgebra::Vector2;
use std::{
	collections::HashSet,
	ffi::{CStr, CString},
	mem::size_of,
	slice,
	sync::Arc,
};
use vk_mem::{Allocation, Allocator, AllocatorCreateInfo};

pub struct Gfx {
	_entry: Entry,
	instance: Instance,
	khr_surface: khr::Surface,
	#[cfg(windows)]
	khr_win32_surface: khr::Win32Surface,
	#[cfg(unix)]
	khr_xlib_surface: khr::XlibSurface,
	#[cfg(unix)]
	khr_wayland_surface: khr::WaylandSurface,
	physical_device: vk::PhysicalDevice,
	queue_family: u32,
	device: Device,
	khr_swapchain: khr::Swapchain,
	queue: vk::Queue,
	cmdpool: vk::CommandPool,
	cmdpool_transient: vk::CommandPool,
	desc_layout: vk::DescriptorSetLayout,
	pipeline_layout: vk::PipelineLayout,
	allocator: Allocator,
	triangle: vk::Buffer,
	triangle_alloc: Allocation,
	sampler: vk::Sampler,
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

		let mut exts = vec![b"VK_KHR_surface\0".as_ptr() as _];
		#[cfg(windows)]
		exts.push(b"VK_KHR_win32_surface\0".as_ptr() as _);
		#[cfg(unix)]
		exts.push(b"VK_KHR_xlib_surface\0".as_ptr() as _);

		let ci = vk::InstanceCreateInfo::builder().application_info(&app_info).enabled_extension_names(&exts);
		let instance = unsafe { entry.create_instance(&ci, None) }.unwrap();
		let khr_surface = khr::Surface::new(&entry, &instance);
		#[cfg(windows)]
		let khr_win32_surface = khr::Win32Surface::new(&entry, &instance);
		#[cfg(unix)]
		let khr_xlib_surface = khr::XlibSurface::new(&entry, &instance);
		#[cfg(unix)]
		let khr_wayland_surface = khr::WaylandSurface::new(&entry, &instance);

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

		let ci = vk::SamplerCreateInfo::builder()
			.mag_filter(vk::Filter::LINEAR)
			.min_filter(vk::Filter::LINEAR)
			.address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
			.address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
			.address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
		let sampler = unsafe { device.create_sampler(&ci, None) }.unwrap();

		let bindings = [
			vk::DescriptorSetLayoutBinding::builder()
				.binding(0)
				.descriptor_type(vk::DescriptorType::SAMPLER)
				.stage_flags(vk::ShaderStageFlags::FRAGMENT)
				.immutable_samplers(&[sampler])
				.build(),
			vk::DescriptorSetLayoutBinding::builder()
				.binding(1)
				.descriptor_count(2)
				.descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
				.stage_flags(vk::ShaderStageFlags::FRAGMENT)
				.build(),
		];
		let ci = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
		let desc_layout = unsafe { device.create_descriptor_set_layout(&ci, None) }.unwrap();

		let desc_layouts = [desc_layout];
		let ci = vk::PipelineLayoutCreateInfo::builder().set_layouts(&desc_layouts);
		let pipeline_layout = unsafe { device.create_pipeline_layout(&ci, None) }.unwrap();

		let ci = AllocatorCreateInfo {
			physical_device,
			device: device.clone(),
			instance: instance.clone(),
			..AllocatorCreateInfo::default()
		};
		let allocator = Allocator::new(&ci).unwrap();

		let verts =
			[TriangleVertex { pos: [-1.0, -1.0].into() }, TriangleVertex { pos: [3.0, -1.0].into() }, TriangleVertex {
				pos: [-1.0, 3.0].into(),
			}];
		let (triangle, triangle_alloc) = create_device_local_buffer(
			&device,
			queue,
			&allocator,
			cmdpool_transient,
			&verts,
			vk::BufferUsageFlags::VERTEX_BUFFER,
		);

		let vshader = create_shader(&device, &vert_spv.await.unwrap());
		let fshader = create_shader(&device, &frag_spv.await.unwrap());

		Arc::new(Self {
			_entry: entry,
			instance,
			khr_surface,
			#[cfg(windows)]
			khr_win32_surface,
			#[cfg(unix)]
			khr_xlib_surface,
			#[cfg(unix)]
			khr_wayland_surface,
			physical_device,
			queue_family,
			device,
			khr_swapchain,
			queue,
			cmdpool,
			cmdpool_transient,
			desc_layout,
			pipeline_layout,
			allocator,
			triangle,
			triangle_alloc,
			sampler,
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
			self.device.destroy_buffer(self.triangle, None);
			self.allocator.free_memory(&self.triangle_alloc).unwrap();
			self.allocator.destroy();
			self.device.destroy_pipeline_layout(self.pipeline_layout, None);
			self.device.destroy_descriptor_set_layout(self.desc_layout, None);
			self.device.destroy_sampler(self.sampler, None);
			self.device.destroy_command_pool(self.cmdpool_transient, None);
			self.device.destroy_command_pool(self.cmdpool, None);
			self.device.destroy_device(None);
			self.instance.destroy_instance(None);
		}
	}
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct TriangleVertex {
	pub pos: Vector2<f32>,
}
impl TriangleVertex {
	fn binding_desc() -> vk::VertexInputBindingDescription {
		vk::VertexInputBindingDescription::builder()
			.binding(0)
			.stride(size_of::<Self>() as _)
			.input_rate(vk::VertexInputRate::VERTEX)
			.build()
	}

	fn attribute_descs() -> [vk::VertexInputAttributeDescription; 1] {
		[vk::VertexInputAttributeDescription::builder()
			.binding(0)
			.location(0)
			.format(vk::Format::R32G32_SFLOAT)
			.offset(offset_of!(Self, pos) as _)
			.build()]
	}
}

fn create_shader(device: &Device, code: &[u8]) -> vk::ShaderModule {
	let code = unsafe { slice::from_raw_parts(code.as_ptr() as _, code.len() / 4) };
	let ci = vk::ShaderModuleCreateInfo::builder().code(code);
	unsafe { device.create_shader_module(&ci, None) }.unwrap()
}
