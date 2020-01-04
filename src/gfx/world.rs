use crate::gfx::{image::create_device_local_image, Gfx};
use ash::{version::DeviceV1_0, vk};
use std::{i8, sync::Arc};
use vk_mem::Allocation;

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
		let mut voxels = vec![0; 16 * 16 * 256];
		for x in 0..16 {
			for y in 0..16 {
				for z in 0..256 {
					voxels[x * y * z + y * z + z] = z;
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
			vk::Format::R8_SNORM,
			vk::Extent3D::builder().width(16).height(16).depth(256).build(),
			vk::ImageUsageFlags::SAMPLED,
		);
		let (mats, mats_alloc, mats_view) = create_device_local_image(
			&gfx.device,
			gfx.queue,
			&gfx.allocator,
			gfx.cmdpool_transient,
			&[0u8; 16 * 16 * 256],
			vk::ImageType::TYPE_3D,
			vk::Format::R8_UINT,
			vk::Extent3D::builder().width(16).height(16).depth(256).build(),
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
