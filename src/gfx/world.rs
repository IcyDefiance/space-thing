use crate::gfx::{image::create_device_local_image, Gfx};
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;
use vk_mem::Allocation;

const res: usize = 4;
const range: f32 = 10.0;


pub struct World {
	gfx: Arc<Gfx>,
	voxels: vk::Image,
	voxels_alloc: Allocation,
	pub voxels_view: vk::ImageView,
	mats: vk::Image,
	mats_alloc: Allocation,
	pub mats_view: vk::ImageView,
}
impl World {
	pub fn new(gfx: Arc<Gfx>) -> Self {
		let mut voxels = vec![0; 16 * 16 * 256 * res * res * res];
		for z in 0..(256 * res) {
			for y in 0..(16 * res) {
				for x in 0..(16 * res) {
					let px = (x as f32)/(res as f32) - 8.0;
					let py = (y as f32)/(res as f32) - 8.0;
					let pz = (z as f32)/(res as f32) - 128.0;
					let mut sd = pz;

					// sphere
					let mut cd = (px*px + py*py + (pz - 3.5)*(pz - 3.5)).sqrt() - 0.5;
					if cd < sd {sd = cd;}

					// box
					let mut qx = px.abs() - 0.5;
					let mut qy = py.abs() - 0.5;
					let mut qz = (pz - 0.5).abs() - 0.5;
					cd = (qx*qx.max(0.0) + qy*qy.max(0.0) + qz*qz.max(0.0)).sqrt() + qx.max(qy.max(qz)).min(0.0);
					if cd < sd {sd = cd;}

					// box
					qx = px.abs() - 0.5 * 0.618;
					qy = py.abs() - 0.5 * 0.618;
					qz = (pz - 1.5).abs() - 0.5 * 0.618;
					cd = (qx*qx.max(0.0) + qy*qy.max(0.0) + qz*qz.max(0.0)).sqrt() + qx.max(qy.max(qz)).min(0.0);
					if cd < sd {sd = cd;}

					// box
					qx = px.abs() - 0.5;
					qy = py.abs() - 0.5;
					qz = (pz - 2.5).abs() - 0.5;
					cd = (qx*qx.max(0.0) + qy*qy.max(0.0) + qz*qz.max(0.0)).sqrt() + qx.max(qy.max(qz)).min(0.0);
					if cd < sd {sd = cd;}

					let mut d = 255.0 * (sd + 1.0) / (range + 1.0);
					voxels[x + y*16*res + z*16*16*res*res] = (d.round() as i64).max(0).min(255) as u8;
				}
			}
		}

		let (voxels, voxels_alloc, voxels_view) = create_device_local_image(
			&gfx.device,
			gfx.queue,
			&gfx.allocator,
			gfx.cmdpool_transient,
			&voxels,
			vk::ImageType::TYPE_3D,
			vk::Format::R8_UNORM,
			vk::Extent3D::builder().width(16 * res as u32).height(16 * res as u32).depth(256 * res as u32).build(),
			vk::ImageUsageFlags::SAMPLED,
		);
		let (mats, mats_alloc, mats_view) = create_device_local_image(
			&gfx.device,
			gfx.queue,
			&gfx.allocator,
			gfx.cmdpool_transient,
			&[0u8; 16 * 16 * 256 * res * res * res],
			vk::ImageType::TYPE_3D,
			vk::Format::R8_UINT,
			vk::Extent3D::builder().width(16 * res as u32).height(16 * res as u32).depth(256 * res as u32).build(),
			vk::ImageUsageFlags::SAMPLED,
		);

		Self { gfx, voxels, voxels_alloc, voxels_view, mats, mats_alloc, mats_view }
	}
}
impl Drop for World {
	fn drop(&mut self) {
		unsafe {
			self.gfx.device.destroy_image_view(self.voxels_view, None);
			self.gfx.device.destroy_image(self.voxels, None);
			self.gfx.allocator.free_memory(&self.voxels_alloc).unwrap();

			self.gfx.device.destroy_image_view(self.mats_view, None);
			self.gfx.device.destroy_image(self.mats, None);
			self.gfx.allocator.free_memory(&self.mats_alloc).unwrap();
		}
	}
}
