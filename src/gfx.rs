pub mod buffer;
pub mod camera;
pub mod image;
pub mod math;
pub mod window;
pub mod world;

use crate::{
	fs::read_bytes,
	gfx::{buffer::create_cpu_buffer, camera::Camera, image::create_device_local_image},
};
use ash::{
	extensions::khr,
	version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
	vk, vk_make_version, Device, Entry, Instance,
};
use buffer::create_device_local_buffer;
use memoffset::offset_of;
use nalgebra::{Vector2, Vector3};
use std::{
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
	gfx_desc_layout: vk::DescriptorSetLayout,
	world_desc_layout: vk::DescriptorSetLayout,
	pipeline_layout: vk::PipelineLayout,
	allocator: Allocator,
	triangle: vk::Buffer,
	triangle_alloc: Allocation,
	voxels_sampler: vk::Sampler,
	mats_sampler: vk::Sampler,
	blocks_sampler: vk::Sampler,
	vshader: vk::ShaderModule,
	fshader: vk::ShaderModule,
	stencil_shader: vk::ShaderModule,
	stencil_desc_layout: vk::DescriptorSetLayout,
	stencil_pipeline_layout: vk::PipelineLayout,
	stencil_pipeline: vk::Pipeline,
	blocks: vk::Image,
	blocks_alloc: Allocation,
	blocks_view: vk::ImageView,
	desc_pool: vk::DescriptorPool,
	desc_set: vk::DescriptorSet,
}
impl Gfx {
	pub async fn new() -> Arc<Self> {
		// start reading files now to use later
		let vert_spv = read_bytes("build/shader.vert.spv");
		let frag_spv = read_bytes("build/shader.frag.spv");
		let stencil_spv = read_bytes("build/stencil.comp.spv");
		let blocks_data = read_bytes("assets/textures.layer1.data");

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
		let voxels_sampler = unsafe { device.create_sampler(&ci, None) }.unwrap();
		let ci = vk::SamplerCreateInfo::builder()
			.mag_filter(vk::Filter::NEAREST)
			.min_filter(vk::Filter::NEAREST)
			.address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
			.address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
			.address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
		let mats_sampler = unsafe { device.create_sampler(&ci, None) }.unwrap();
		let ci = vk::SamplerCreateInfo::builder()
			.mag_filter(vk::Filter::LINEAR)
			.min_filter(vk::Filter::LINEAR)
			.address_mode_u(vk::SamplerAddressMode::REPEAT)
			.address_mode_v(vk::SamplerAddressMode::REPEAT)
			.address_mode_w(vk::SamplerAddressMode::REPEAT);
		let blocks_sampler = unsafe { device.create_sampler(&ci, None) }.unwrap();

		let bindings = [vk::DescriptorSetLayoutBinding::builder()
			.binding(0)
			.descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
			.stage_flags(vk::ShaderStageFlags::FRAGMENT)
			.immutable_samplers(&[blocks_sampler])
			.build()];
		let ci = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
		let gfx_desc_layout = unsafe { device.create_descriptor_set_layout(&ci, None) }.unwrap();
		let bindings = [
			vk::DescriptorSetLayoutBinding::builder()
				.binding(0)
				.descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
				.stage_flags(vk::ShaderStageFlags::FRAGMENT)
				.immutable_samplers(&[voxels_sampler])
				.build(),
			vk::DescriptorSetLayoutBinding::builder()
				.binding(1)
				.descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
				.stage_flags(vk::ShaderStageFlags::FRAGMENT)
				.immutable_samplers(&[mats_sampler])
				.build(),
		];
		let ci = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
		let world_desc_layout = unsafe { device.create_descriptor_set_layout(&ci, None) }.unwrap();

		let desc_layouts = [gfx_desc_layout, world_desc_layout];
		let push_constant_ranges = [vk::PushConstantRange::builder()
			.stage_flags(vk::ShaderStageFlags::FRAGMENT)
			.offset(0)
			.size(size_of::<Camera>() as _)
			.build()];
		let ci = vk::PipelineLayoutCreateInfo::builder()
			.set_layouts(&desc_layouts)
			.push_constant_ranges(&push_constant_ranges);
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
		let stencil_shader = create_shader(&device, &stencil_spv.await.unwrap());

		let (stencil_desc_layout, stencil_pipeline_layout, stencil_pipeline) =
			create_stencil_pipeline(&device, stencil_shader);

		let blocks_data = blocks_data.await.unwrap();
		let (blocks_cpu, blocks_cpualloc, blocks_cpumap) = create_cpu_buffer::<u8>(&allocator, blocks_data.len());
		blocks_cpumap.copy_from_slice(&blocks_data);
		let (blocks, blocks_alloc, blocks_view) = create_device_local_image(
			&device,
			queue,
			&allocator,
			cmdpool_transient,
			vk::ImageType::TYPE_2D,
			vk::Format::R8G8B8A8_SRGB,
			vk::Extent3D::builder().width(4096).height(512).depth(1).build(),
			true,
			vk::ImageUsageFlags::SAMPLED,
			blocks_cpu,
		);
		unsafe { device.destroy_buffer(blocks_cpu, None) };
		allocator.free_memory(&blocks_cpualloc).unwrap();

		let (desc_pool, desc_set) = create_desc_pool(&device, gfx_desc_layout, blocks_view);

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
			gfx_desc_layout,
			world_desc_layout,
			pipeline_layout,
			allocator,
			triangle,
			triangle_alloc,
			voxels_sampler,
			mats_sampler,
			blocks_sampler,
			vshader,
			fshader,
			stencil_shader,
			stencil_desc_layout,
			stencil_pipeline_layout,
			stencil_pipeline,
			blocks,
			blocks_alloc,
			blocks_view,
			desc_pool,
			desc_set,
		})
	}
}
impl Drop for Gfx {
	fn drop(&mut self) {
		unsafe {
			self.device.destroy_image_view(self.blocks_view, None);
			self.device.destroy_image(self.blocks, None);
			self.allocator.free_memory(&self.blocks_alloc).unwrap();
			self.device.destroy_pipeline(self.stencil_pipeline, None);
			self.device.destroy_pipeline_layout(self.stencil_pipeline_layout, None);
			self.device.destroy_shader_module(self.stencil_shader, None);
			self.device.destroy_shader_module(self.fshader, None);
			self.device.destroy_shader_module(self.vshader, None);
			self.device.destroy_buffer(self.triangle, None);
			self.allocator.free_memory(&self.triangle_alloc).unwrap();
			self.allocator.destroy();
			self.device.destroy_pipeline_layout(self.pipeline_layout, None);
			self.device.destroy_descriptor_set_layout(self.stencil_desc_layout, None);
			self.device.destroy_descriptor_set_layout(self.world_desc_layout, None);
			self.device.destroy_descriptor_set_layout(self.gfx_desc_layout, None);
			self.device.destroy_sampler(self.blocks_sampler, None);
			self.device.destroy_sampler(self.mats_sampler, None);
			self.device.destroy_sampler(self.voxels_sampler, None);
			self.device.destroy_command_pool(self.cmdpool_transient, None);
			self.device.destroy_command_pool(self.cmdpool, None);
			self.device.destroy_descriptor_pool(self.desc_pool, None);
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

fn create_stencil_pipeline(
	device: &Device,
	module: vk::ShaderModule,
) -> (vk::DescriptorSetLayout, vk::PipelineLayout, vk::Pipeline) {
	let bindings = [vk::DescriptorSetLayoutBinding::builder()
		.binding(0)
		.descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
		.descriptor_count(1)
		.stage_flags(vk::ShaderStageFlags::COMPUTE)
		.build()];
	let ci = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
	let desc_layout = unsafe { device.create_descriptor_set_layout(&ci, None) }.unwrap();

	let set_layouts = [desc_layout];
	let push_constant_ranges = [vk::PushConstantRange::builder()
		.stage_flags(vk::ShaderStageFlags::COMPUTE)
		.offset(0)
		.size(size_of::<Vector3<f32>>() as _)
		.build()];
	let ci =
		vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts).push_constant_ranges(&push_constant_ranges);
	let layout = unsafe { device.create_pipeline_layout(&ci, None) }.unwrap();

	let name = CStr::from_bytes_with_nul(b"main\0").unwrap();
	let stage = vk::PipelineShaderStageCreateInfo::builder()
		.stage(vk::ShaderStageFlags::COMPUTE)
		.module(module)
		.name(name)
		.build();
	let ci = vk::ComputePipelineCreateInfo::builder().stage(stage).layout(layout).build();
	let pipeline = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[ci], None) }.unwrap()[0];

	(desc_layout, layout, pipeline)
}

fn create_shader(device: &Device, code: &[u8]) -> vk::ShaderModule {
	let code = unsafe { slice::from_raw_parts(code.as_ptr() as _, code.len() / 4) };
	let ci = vk::ShaderModuleCreateInfo::builder().code(code);
	unsafe { device.create_shader_module(&ci, None) }.unwrap()
}

fn create_desc_pool(
	device: &Device,
	desc_layout: vk::DescriptorSetLayout,
	blocks_view: vk::ImageView,
) -> (vk::DescriptorPool, vk::DescriptorSet) {
	let pool_sizes =
		[vk::DescriptorPoolSize::builder().ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(3).build()];
	let ci = vk::DescriptorPoolCreateInfo::builder().max_sets(1).pool_sizes(&pool_sizes);
	let desc_pool = unsafe { device.create_descriptor_pool(&ci, None) }.unwrap();

	let set_layouts = [desc_layout];
	let ci = vk::DescriptorSetAllocateInfo::builder().descriptor_pool(desc_pool).set_layouts(&set_layouts);
	let desc_set = unsafe { device.allocate_descriptor_sets(&ci) }.unwrap()[0];

	let blocks_info = [vk::DescriptorImageInfo::builder()
		.image_view(blocks_view)
		.image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
		.build()];
	let write = [vk::WriteDescriptorSet::builder()
		.dst_set(desc_set)
		.dst_binding(0)
		.descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
		.image_info(&blocks_info)
		.build()];
	unsafe { device.update_descriptor_sets(&write, &[]) };

	(desc_pool, desc_set)
}
