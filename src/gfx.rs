pub mod gui;
pub mod volume;
pub mod window;

use crate::fs::read_all_u32;
use ash::vk;
use memoffset::offset_of;
use nalgebra::Vector2;
#[cfg(debug_assertions)]
use std::ffi::CString;
use std::{mem::size_of, sync::Arc};
use typenum::{B0, B1};
use vulkan::{
	buffer::Buffer,
	device::{BufferUsageFlags, Device, Queue},
	instance::{Instance, Version},
	pipeline::PipelineLayout,
	shader::ShaderModule,
	Vulkan,
};

pub struct Gfx {
	instance: Arc<Instance>,
	device: Arc<Device>,
	queue: Arc<Queue>,
	layout: PipelineLayout,
	triangle: Arc<Buffer<[TriangleVertex]>>,
	vshader: ShaderModule,
	fshader: ShaderModule,
}
impl Gfx {
	pub async fn new() -> Arc<Self> {
		// start reading files now to use later
		let vert_spv = read_all_u32("build/shader.vert.spv");
		let frag_spv = read_all_u32("build/shader.frag.spv");

		let vulkan = Vulkan::new().unwrap();

		let name = CString::new(env!("CARGO_PKG_NAME")).unwrap();
		let version = Version::new(
			env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap(),
			env!("CARGO_PKG_VERSION_MINOR").parse().unwrap(),
			env!("CARGO_PKG_VERSION_PATCH").parse().unwrap(),
		);
		let instance = Instance::new(vulkan, &name, version);

		let (device, mut queue) = {
			let physical_device = instance.enumerate_physical_devices().next().unwrap();

			let queue_family = physical_device
				.get_queue_family_properties()
				.filter(|props| props.queue_flags().graphics())
				.next()
				.unwrap()
				.family();

			let (device, mut queues) = physical_device.create_device(vec![(queue_family, &[1.0][..])]);
			(device, queues.next().unwrap())
		};

		let layout = device.create_pipeline_layout();

		let cmdpool = device.create_command_pool(queue.family(), true);

		let verts =
			[TriangleVertex { pos: [-1.0, -1.0].into() }, TriangleVertex { pos: [3.0, -1.0].into() }, TriangleVertex {
				pos: [-1.0, 3.0].into(),
			}];
		let triangle =
			device.create_buffer_slice(verts.len() as _, B1, BufferUsageFlags::TRANSFER_SRC).copy_from_slice(&verts);
		let (triangle, future) = device
			.create_buffer_slice(verts.len() as _, B0, BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::VERTEX_BUFFER)
			.copy_from_buffer(&mut queue, &cmdpool, triangle);
		let mut fence = device.create_fence(false);
		future.end(&mut fence);
		fence.wait();

		let vshader = unsafe { device.create_shader_module(&vert_spv.await.unwrap()) };
		let fshader = unsafe { device.create_shader_module(&frag_spv.await.unwrap()) };

		Arc::new(Self { instance, device, queue, layout, triangle, vshader, fshader })
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
