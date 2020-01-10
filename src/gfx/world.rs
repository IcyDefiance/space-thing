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

const RES: usize = 4;
const RANGE: f32 = 10.0;

pub struct World {
	gfx: Arc<Gfx>,

	voxels_cpu: vk::Buffer,
	voxels_cpualloc: Allocation,
	voxels_cpumap: &'static mut [[[u8; 256]; 16]; 16],
	pub(super) voxels: vk::Image,
	voxels_alloc: Allocation,
	pub(super) voxels_view: vk::ImageView,

	mats_cpu: vk::Buffer,
	mats_cpualloc: Allocation,
	mats_cpumap: &'static mut [[[u8; 256]; 16]; 16],
	pub(super) mats: vk::Image,
	mats_alloc: Allocation,
	pub(super) mats_view: vk::ImageView,

	desc_pool: vk::DescriptorPool,
	pub(super) desc_set: vk::DescriptorSet,

	pub(super) set_cmds: Vec<Vector3<u32>>,
}
impl World {
	pub fn new(gfx: Arc<Gfx>) -> Self {
		let res32 = RES as u32;
		let res64 = RES as u64;
		let chunk_size = 16 * 16 * 256 * res64 * res64 * res64;
		let chunk_extent = vk::Extent3D { width: 16 * res32, height: 16 * res32, depth: 256 * res32 };

		let (voxels_cpu, voxels_cpualloc, voxels_cpumap) = create_cpu_buffer::<u8>(&gfx.allocator, chunk_size as _);
		init_voxels(voxels_cpumap);
		let (voxels, voxels_alloc, voxels_view) = create_device_local_image(
			&gfx.device,
			gfx.queue,
			&gfx.allocator,
			gfx.cmdpool_transient,
			vk::ImageType::TYPE_3D,
			vk::Format::R8_UNORM,
			chunk_extent,
			false,
			vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE,
			voxels_cpu,
		);

		let (mats_cpu, mats_cpualloc, mats_cpumap) = create_cpu_buffer::<u8>(&gfx.allocator, chunk_size as _);
		unsafe { write_bytes(mats_cpumap.as_mut_ptr(), 1, mats_cpumap.len()) };
		let (mats, mats_alloc, mats_view) = create_device_local_image(
			&gfx.device,
			gfx.queue,
			&gfx.allocator,
			gfx.cmdpool_transient,
			vk::ImageType::TYPE_3D,
			vk::Format::R8_UNORM,
			chunk_extent,
			false,
			vk::ImageUsageFlags::SAMPLED,
			mats_cpu,
		);

		let (desc_pool, desc_set) = create_desc_pool(&gfx, voxels_view, mats_view);

		Self {
			gfx,
			voxels_cpu,
			voxels_cpualloc,
			voxels_cpumap: unsafe { transmute(voxels_cpumap.as_mut_ptr()) },
			voxels,
			voxels_alloc,
			voxels_view,
			mats_cpu,
			mats_cpualloc,
			mats_cpumap: unsafe { transmute(mats_cpumap.as_mut_ptr()) },
			mats,
			mats_alloc,
			mats_view,
			desc_pool,
			desc_set,
			set_cmds: vec![],
		}
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
		(RANGE + 1.0) * (self.voxels_cpumap[x][y][z] as f32) / 255.0 - 1.0
	}
}
impl Drop for World {
	fn drop(&mut self) {
		unsafe {
			self.gfx.device.destroy_buffer(self.voxels_cpu, None);
			self.gfx.allocator.free_memory(&self.voxels_cpualloc).unwrap();
			self.gfx.device.destroy_image_view(self.voxels_view, None);
			self.gfx.device.destroy_image(self.voxels, None);
			self.gfx.allocator.free_memory(&self.voxels_alloc).unwrap();

			self.gfx.device.destroy_buffer(self.mats_cpu, None);
			self.gfx.allocator.free_memory(&self.mats_cpualloc).unwrap();
			self.gfx.device.destroy_image_view(self.mats_view, None);
			self.gfx.device.destroy_image(self.mats, None);
			self.gfx.allocator.free_memory(&self.mats_alloc).unwrap();
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

	for z in 0..(256 * RES) {
		for y in 0..(16 * RES) {
			for x in 0..(16 * RES) {
				let px = x as f32 / resf;
				let py = y as f32 / resf;
				let pz = z as f32 / resf;

				let mut sd = pz - 1.0;
				sd = sd.min(sd_cube(px - 5.5, py - 2.5, pz - 8.5));
				sd = sd.min(sd_cube(px - 3.5, py - 2.5, pz - 8.5));
				sd = sd.min(sd_cube(px - 2.5, py - 2.5, pz - 8.5));
				sd = sd.min(sd_cube(px - 3.5, py - 3.5, pz - 8.5));

				let d = 255.0 * (sd + 1.0) / (RANGE + 1.0);
				voxels[x + y * 16 * RES + z * 16 * 16 * RES * RES] = (d.round() as i64).max(0).min(255) as u8;
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
