use crate::gfx::Gfx;
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;
use vk_mem::{Allocation, AllocationCreateInfo, MemoryUsage};

pub struct Volume {
	image: vk::Image,
	allocation: Allocation,
}
impl Volume {
	pub fn new(gfx: Arc<Gfx>) -> Self {
		#[rustfmt::skip]
		let data = [
			53i8, 0, 53,
			0, 127, 0,
			53i8, 0, 53,

			53i8, 0, 53,
			0, 127, 0,
			53i8, 0, 53,

			53i8, 0, 53,
			0, 127, 0,
			53i8, 0, 53,
		];

		let ci = vk::ImageCreateInfo::builder()
			.image_type(vk::ImageType::TYPE_3D)
			.format(vk::Format::R8_SNORM)
			.extent(vk::Extent3D { width: 3, height: 3, depth: 3 })
			.mip_levels(1)
			.array_layers(1)
			.samples(vk::SampleCountFlags::TYPE_1)
			.usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
			.sharing_mode(vk::SharingMode::EXCLUSIVE)
			.initial_layout(vk::ImageLayout::UNDEFINED);
		let aci = AllocationCreateInfo { usage: MemoryUsage::GpuOnly, ..Default::default() };
		let (image, allocation, _) = gfx.device.allocator.create_image(&ci, &aci).unwrap();

		Self { image, allocation }
	}
}
