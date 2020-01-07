use crate::gfx::{buffer::create_cpu_buffer, image::create_device_local_image, Gfx};
use ash::{version::DeviceV1_0, vk};
use std::{ptr::write_bytes, sync::Arc};
use vk_mem::Allocation;

const RES: usize = 4;
const RANGE: f32 = 10.0;

pub struct World {
	gfx: Arc<Gfx>,

	voxels_cpu: vk::Buffer,
	voxels_cpualloc: Allocation,
	voxels_cpumap: &'static mut [u8],
	voxels: vk::Image,
	voxels_alloc: Allocation,
	pub voxels_view: vk::ImageView,

	mats_cpu: vk::Buffer,
	mats_cpualloc: Allocation,
	mats_cpumap: &'static mut [u8],
	mats: vk::Image,
	mats_alloc: Allocation,
	pub mats_view: vk::ImageView,
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
			vk::ImageUsageFlags::SAMPLED,
			voxels_cpu,
		);

		let (mats_cpu, mats_cpualloc, mats_cpumap) = create_cpu_buffer::<u8>(&gfx.allocator, chunk_size as _);
		unsafe { write_bytes(mats_cpumap.as_mut_ptr(), 0, mats_cpumap.len()) };
		let (mats, mats_alloc, mats_view) = create_device_local_image(
			&gfx.device,
			gfx.queue,
			&gfx.allocator,
			gfx.cmdpool_transient,
			vk::ImageType::TYPE_3D,
			vk::Format::R8_UINT,
			chunk_extent,
			vk::ImageUsageFlags::SAMPLED,
			mats_cpu,
		);

		Self {
			gfx,
			voxels_cpu,
			voxels_cpualloc,
			voxels_cpumap,
			voxels,
			voxels_alloc,
			voxels_view,
			mats_cpu,
			mats_cpualloc,
			mats_cpumap,
			mats,
			mats_alloc,
			mats_view,
		}
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

fn sdCube(x: f32, y: f32, z: f32) -> f32 {
	let qx = x.abs() - 0.5;
    let qy = y.abs() - 0.5;
    let qz = z.abs() - 0.5;
    let l2 = qx*qx.max(0.0) + qy*qy.max(0.0) + qz*qz.max(0.0);
    return l2.sqrt() + qx.max(qy.max(qz)).min(0.0);
}

fn init_voxels(voxels: &mut [u8]) {
	let resf = RES as f32;

	for z in 0..(256 * RES) {
		for y in 0..(16 * RES) {
			for x in 0..(16 * RES) {
				let mut px = (x as f32) / resf;
				let mut py = (y as f32) / resf;
				let mut pz = (z as f32) / resf;

				let mut sd = pz - 1.0;
				sd = sd.min(sdCube(px-5.5, py-2.5, pz-8.5));
				sd = sd.min(sdCube(px-3.5, py-2.5, pz-8.5));
				sd = sd.min(sdCube(px-2.5, py-2.5, pz-8.5));
				sd = sd.min(sdCube(px-3.5, py-3.5, pz-8.5));

				let d = 255.0 * (sd + 1.0) / (RANGE + 1.0);
				voxels[x + y * 16 * RES + z * 16 * 16 * RES * RES] = (d.round() as i64).max(0).min(255) as u8;
			}
		}
	}
}
