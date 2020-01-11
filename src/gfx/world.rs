use crate::gfx::{
	buffer::create_cpu_buffer,
	image::create_device_local_image,
	math::{lerp, v3max},
	Gfx,
};
use array_init::array_init;
use ash::{version::DeviceV1_0, vk};
use nalgebra::{zero, Vector2, Vector3};
use std::{
	alloc::{alloc, alloc_zeroed, Layout},
	mem::MaybeUninit,
	sync::Arc,
};
use vk_mem::Allocation;

const RES: usize = 4;
const RANGE: f32 = 10.0;
const CHUNK_EXTENT: vk::Extent3D =
	vk::Extent3D { width: 16 * RES as u32, height: 16 * RES as u32, depth: 256 * RES as u32 };

type ChunkData = [[[u8; 16 * RES]; 16 * RES]; 256 * RES];
type ChunkArray = [[ChunkLayer; 21]; 21];

pub struct World {
	gfx: Arc<Gfx>,

	pub(super) sdfs: ChunkArray,
	pub(super) mats: ChunkArray,
	pub(super) off: Vector2<u8>,

	desc_pool: vk::DescriptorPool,
	pub(super) desc_set: vk::DescriptorSet,
	stencil_desc_pool: vk::DescriptorPool,
	pub(super) stencil_desc_set: vk::DescriptorSet,

	pub(super) set_cmds: Vec<Vector3<u32>>,
}
impl World {
	pub fn new(gfx: Arc<Gfx>) -> Self {
		let mut sdf = unsafe { Box::from_raw(alloc(Layout::new::<ChunkData>()) as _) };
		init_sdf(&mut *sdf);
		let sdfs: ChunkArray =
			array_init(|_| array_init(|_| ChunkLayer::new(gfx.clone(), vk::ImageUsageFlags::STORAGE, sdf.clone())));

		let mats: Box<ChunkData> = unsafe { Box::from_raw(alloc_zeroed(Layout::new::<ChunkData>()) as _) };
		let mats: ChunkArray =
			array_init(|_| array_init(|_| ChunkLayer::new(gfx.clone(), vk::ImageUsageFlags::empty(), mats.clone())));

		let off = zero();

		let (desc_pool, desc_set) = create_desc_pool(
			&gfx,
			sdfs.iter().map(|x| x.iter().map(|x| x.view)).flatten(),
			mats.iter().map(|x| x.iter().map(|x| x.view)).flatten(),
		);
		let (stencil_desc_pool, stencil_desc_set) =
			create_stencil_desc_pool(&gfx, sdfs.iter().map(|x| x.iter().map(|x| x.view)).flatten());

		Self { gfx, sdfs, mats, off, desc_pool, desc_set, stencil_desc_pool, stencil_desc_set, set_cmds: vec![] }
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
		let (x, y, z) = (pos.x.floor(), pos.y.floor(), pos.z as usize);
		let (tx, ty, tz) = (pos.x - x, pos.y - y, pos.z - z as f32);

		let c000 = self.sample_exact(x, y, z);
		let c001 = self.sample_exact(x, y, z + 1);
		let c010 = self.sample_exact(x, y + 1.0, z);
		let c011 = self.sample_exact(x, y + 1.0, z + 1);
		let c100 = self.sample_exact(x + 1.0, y, z);
		let c101 = self.sample_exact(x + 1.0, y, z + 1);
		let c110 = self.sample_exact(x + 1.0, y + 1.0, z);
		let c111 = self.sample_exact(x + 1.0, y + 1.0, z + 1);

		let c00 = lerp(c000, c100, tx);
		let c01 = lerp(c001, c101, tx);
		let c10 = lerp(c010, c110, tx);
		let c11 = lerp(c011, c111, tx);

		let c0 = lerp(c00, c10, ty);
		let c1 = lerp(c01, c11, ty);

		lerp(c0, c1, tz)
	}

	fn sample_exact(&self, x: f32, y: f32, z: usize) -> f32 {
		let sdfy = ((y / 16.0).floor() - self.off.y as f32 + 10.0) as usize;
		let sdfx = ((x / 16.0).floor() - self.off.x as f32 + 10.0) as usize;
		let sdf = &self.sdfs[sdfy][sdfx];
		let y = (y % 16.0 + 16.0) as usize;
		let x = (x % 16.0 + 16.0) as usize;
		(RANGE + 1.0) * (sdf.data[z][y][x] as f32) / 255.0 - 1.0
	}
}
impl Drop for World {
	fn drop(&mut self) {
		unsafe { self.gfx.device.destroy_descriptor_pool(self.stencil_desc_pool, None) };
		unsafe { self.gfx.device.destroy_descriptor_pool(self.desc_pool, None) };
	}
}

pub(super) struct ChunkLayer {
	gfx: Arc<Gfx>,
	data: Box<ChunkData>,
	pub(super) image: vk::Image,
	alloc: Allocation,
	pub(super) view: vk::ImageView,
}
impl ChunkLayer {
	fn new(gfx: Arc<Gfx>, usage: vk::ImageUsageFlags, data: Box<ChunkData>) -> Self {
		let (buf, cpualloc, map) = create_cpu_buffer::<[[u8; 16 * RES]; 16 * RES]>(&gfx.allocator, 256 * RES);
		map.copy_from_slice(&*data);
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
		unsafe {
			gfx.device.destroy_buffer(buf, None);
			gfx.allocator.free_memory(&cpualloc).unwrap();
		}

		Self { gfx, data, image, alloc, view }
	}
}
impl Drop for ChunkLayer {
	fn drop(&mut self) {
		unsafe {
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

fn init_sdf(voxels: &mut [[[u8; 16 * RES]; 16 * RES]; 256 * RES]) {
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
				voxels[z][y][x] = (d.round() as i64).max(0).min(255) as u8;
			}
		}
	}
}

fn create_desc_pool(
	gfx: &Gfx,
	voxel_views: impl IntoIterator<Item = vk::ImageView>,
	mat_views: impl IntoIterator<Item = vk::ImageView>,
) -> (vk::DescriptorPool, vk::DescriptorSet) {
	let pool_sizes = [
		vk::DescriptorPoolSize::builder().ty(vk::DescriptorType::SAMPLER).descriptor_count(2).build(),
		vk::DescriptorPoolSize::builder().ty(vk::DescriptorType::SAMPLED_IMAGE).descriptor_count(441).build(),
		vk::DescriptorPoolSize::builder().ty(vk::DescriptorType::SAMPLED_IMAGE).descriptor_count(441).build(),
	];
	let ci = vk::DescriptorPoolCreateInfo::builder().max_sets(1).pool_sizes(&pool_sizes);
	let desc_pool = unsafe { gfx.device.create_descriptor_pool(&ci, None) }.unwrap();

	let set_layouts = [gfx.world_desc_layout];
	let ci = vk::DescriptorSetAllocateInfo::builder().descriptor_pool(desc_pool).set_layouts(&set_layouts);
	let desc_set = unsafe { gfx.device.allocate_descriptor_sets(&ci) }.unwrap()[0];

	let voxel_infos: Vec<_> = voxel_views
		.into_iter()
		.map(|image_view| {
			vk::DescriptorImageInfo::builder()
				.image_view(image_view)
				.image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
				.build()
		})
		.collect();
	let mat_infos: Vec<_> = mat_views
		.into_iter()
		.map(|image_view| {
			vk::DescriptorImageInfo::builder()
				.image_view(image_view)
				.image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
				.build()
		})
		.collect();
	let write = [
		vk::WriteDescriptorSet::builder()
			.dst_set(desc_set)
			.dst_binding(1)
			.descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
			.image_info(&voxel_infos)
			.build(),
		vk::WriteDescriptorSet::builder()
			.dst_set(desc_set)
			.dst_binding(2)
			.descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
			.image_info(&mat_infos)
			.build(),
	];
	unsafe { gfx.device.update_descriptor_sets(&write, &[]) };

	(desc_pool, desc_set)
}

fn create_stencil_desc_pool(
	gfx: &Gfx,
	voxel_views: impl IntoIterator<Item = vk::ImageView>,
) -> (vk::DescriptorPool, vk::DescriptorSet) {
	let pool_sizes =
		[vk::DescriptorPoolSize::builder().ty(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(882).build()];
	let ci = vk::DescriptorPoolCreateInfo::builder().max_sets(1).pool_sizes(&pool_sizes);
	let desc_pool = unsafe { gfx.device.create_descriptor_pool(&ci, None) }.unwrap();

	let set_layouts = [gfx.stencil_desc_layout];
	let ci = vk::DescriptorSetAllocateInfo::builder().descriptor_pool(desc_pool).set_layouts(&set_layouts);
	let desc_set = unsafe { gfx.device.allocate_descriptor_sets(&ci) }.unwrap()[0];

	let voxel_infos: Vec<_> = voxel_views
		.into_iter()
		.map(|image_view| {
			vk::DescriptorImageInfo::builder().image_view(image_view).image_layout(vk::ImageLayout::GENERAL).build()
		})
		.collect();
	let write = [vk::WriteDescriptorSet::builder()
		.dst_set(desc_set)
		.dst_binding(0)
		.descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
		.image_info(&voxel_infos)
		.build()];
	unsafe { gfx.device.update_descriptor_sets(&write, &[]) };

	(desc_pool, desc_set)
}
