use crate::gfx::{
	buffer::create_cpu_buffer,
	image::create_device_local_image,
	math::{lerp, v3max},
	Gfx,
};
use ash::{version::DeviceV1_0, vk};
use nalgebra::Vector3;
use std::{mem::transmute, ptr::write_bytes, sync::Arc};
use vk_mem::Allocation;

const RES: u32 = 4;
const RANGE: f32 = 10.0;
const CHUNK_EXTENT: vk::Extent3D = vk::Extent3D { width: 16 * RES, height: 16 * RES, depth: 256 * RES };
const CHUNK_SIZE: usize = (CHUNK_EXTENT.width * CHUNK_EXTENT.height * CHUNK_EXTENT.depth) as _;

pub struct World {
	gfx: Arc<Gfx>,

	pub(super) voxels: ChunkLayer,
	pub(super) mats: ChunkLayer,

	desc_pool: vk::DescriptorPool,
	pub(super) desc_set: vk::DescriptorSet,

	pub(super) set_cmds: Vec<Vector3<u32>>,
}
impl World {
	pub fn new(gfx: Arc<Gfx>) -> Self {
		let voxels = ChunkLayer::new(gfx.clone(), vk::ImageUsageFlags::STORAGE, init_voxels);
		let mats = ChunkLayer::new(gfx.clone(), vk::ImageUsageFlags::empty(), |map| unsafe {
			write_bytes(map.as_mut_ptr(), 1, map.len())
		});

		let (desc_pool, desc_set) = create_desc_pool(&gfx, voxels.view, mats.view);

		Self { gfx, voxels, mats, desc_pool, desc_set, set_cmds: vec![] }
	}

	pub fn set_block(&mut self, pos: Vector3<u32>) {
		self.set_cmds.push(pos);
	}

	/// assumes dir is normalized
	pub fn sphere_sweep(&self, start: Vector3<f32>, dir: Vector3<f32>, len: f32, radius: f32) -> f32 {
		let collide = 0.01;
		let mut dist = 0.0;
		while dist < len {
			let march = self.sample(start + dir * dist) - radius;
			if march < collide {
				break;
			}

			dist += march;
		}
		dist
	}

	fn sample(&self, pos: Vector3<f32>) -> f32 {
		let pos = v3max((pos * 4.0).add_scalar(-0.5), 0.0);
		let (x, y, z) = (pos.x as usize, pos.y as usize, pos.z as usize);
		let (tx, ty, tz) = (pos.x - x as f32, pos.y - y as f32, pos.z - z as f32);

		let c000 = self.sample_exact(x, y, z);
		let c001 = self.sample_exact(x, y, z + 1);
		let c010 = self.sample_exact(x, y + 1, z);
		let c011 = self.sample_exact(x, y + 1, z + 1);
		let c100 = self.sample_exact(x + 1, y, z);
		let c101 = self.sample_exact(x + 1, y, z + 1);
		let c110 = self.sample_exact(x + 1, y + 1, z);
		let c111 = self.sample_exact(x + 1, y + 1, z + 1);

		let c00 = lerp(c000, c100, tx);
		let c01 = lerp(c001, c101, tx);
		let c10 = lerp(c010, c110, tx);
		let c11 = lerp(c011, c111, tx);

		let c0 = lerp(c00, c10, ty);
		let c1 = lerp(c01, c11, ty);

		lerp(c0, c1, tz)
	}

	fn sample_exact(&self, x: usize, y: usize, z: usize) -> f32 {
		(RANGE + 1.0) * (self.voxels.map[x][y][z] as f32) / 255.0 - 1.0
	}
}
impl Drop for World {
	fn drop(&mut self) {
		unsafe { self.gfx.device.destroy_descriptor_pool(self.desc_pool, None) };
	}
}

pub(super) struct ChunkLayer {
	gfx: Arc<Gfx>,
	buf: vk::Buffer,
	cpualloc: Allocation,
	map: &'static mut [[[u8; 256]; 16]; 16],
	pub(super) image: vk::Image,
	alloc: Allocation,
	pub(super) view: vk::ImageView,
}
impl ChunkLayer {
	fn new(gfx: Arc<Gfx>, usage: vk::ImageUsageFlags, init: impl FnOnce(&mut [u8])) -> Self {
		let (buf, cpualloc, map) = create_cpu_buffer::<u8>(&gfx.allocator, CHUNK_SIZE);
		init(map);
		let (image, alloc, view) = create_device_local_image(
			&gfx.device,
			gfx.queue,
			&gfx.allocator,
			gfx.cmdpool_transient,
			vk::ImageType::TYPE_3D,
			vk::Format::R8_UNORM,
			CHUNK_EXTENT,
			false,
			vk::ImageUsageFlags::SAMPLED | usage,
			buf,
		);

		Self { gfx, buf, cpualloc, map: unsafe { transmute(map.as_mut_ptr()) }, image, alloc, view }
	}
}
impl Drop for ChunkLayer {
	fn drop(&mut self) {
		unsafe {
			self.gfx.device.destroy_buffer(self.buf, None);
			self.gfx.allocator.free_memory(&self.cpualloc).unwrap();
			self.gfx.device.destroy_image_view(self.view, None);
			self.gfx.device.destroy_image(self.image, None);
			self.gfx.allocator.free_memory(&self.alloc).unwrap();
		}
	}
}

fn sd_cube(x: f32, y: f32, z: f32) -> f32 {
	let qx = x.abs() - 0.5;
	let qy = y.abs() - 0.5;
	let qz = z.abs() - 0.5;
	let l2 = qx * qx.max(0.0) + qy * qy.max(0.0) + qz * qz.max(0.0);
	return l2.sqrt() + qx.max(qy.max(qz)).min(0.0);
}

fn init_voxels(voxels: &mut [u8]) {
	let resf = RES as f32;
	let resu = RES as usize;

	for z in 0..(256 * resu) {
		for y in 0..(16 * resu) {
			for x in 0..(16 * resu) {
				let px = x as f32 / resf;
				let py = y as f32 / resf;
				let pz = z as f32 / resf;

				let mut sd = pz - 1.0;
				sd = sd.min(sd_cube(px - 5.5, py - 2.5, pz - 8.5));
				sd = sd.min(sd_cube(px - 3.5, py - 2.5, pz - 8.5));
				sd = sd.min(sd_cube(px - 2.5, py - 2.5, pz - 8.5));
				sd = sd.min(sd_cube(px - 3.5, py - 3.5, pz - 8.5));

				let d = 255.0 * (sd + 1.0) / (RANGE + 1.0);
				voxels[x + y * 16 * resu + z * 16 * 16 * resu * resu] = (d.round() as i64).max(0).min(255) as u8;
			}
		}
	}
}

fn create_desc_pool(
	gfx: &Gfx,
	voxels_view: vk::ImageView,
	mats_view: vk::ImageView,
) -> (vk::DescriptorPool, vk::DescriptorSet) {
	let pool_sizes =
		[vk::DescriptorPoolSize::builder().ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(3).build()];
	let ci = vk::DescriptorPoolCreateInfo::builder().max_sets(1).pool_sizes(&pool_sizes);
	let desc_pool = unsafe { gfx.device.create_descriptor_pool(&ci, None) }.unwrap();

	let set_layouts = [gfx.world_desc_layout];
	let ci = vk::DescriptorSetAllocateInfo::builder().descriptor_pool(desc_pool).set_layouts(&set_layouts);
	let desc_set = unsafe { gfx.device.allocate_descriptor_sets(&ci) }.unwrap()[0];

	let voxels_info = [vk::DescriptorImageInfo::builder()
		.image_view(voxels_view)
		.image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
		.build()];
	let mats_info = [vk::DescriptorImageInfo::builder()
		.image_view(mats_view)
		.image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
		.build()];
	let write = [
		vk::WriteDescriptorSet::builder()
			.dst_set(desc_set)
			.dst_binding(0)
			.descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
			.image_info(&voxels_info)
			.build(),
		vk::WriteDescriptorSet::builder()
			.dst_set(desc_set)
			.dst_binding(1)
			.descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
			.image_info(&mats_info)
			.build(),
	];
	unsafe { gfx.device.update_descriptor_sets(&write, &[]) };

	(desc_pool, desc_set)
}
