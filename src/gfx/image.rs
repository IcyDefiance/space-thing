use ash::{version::DeviceV1_0, vk, Device};
use vk_mem::{Allocation, AllocationCreateInfo, Allocator, MemoryUsage};

pub(super) fn create_device_local_image(
	device: &Device,
	queue: vk::Queue,
	allocator: &Allocator,
	cmdpool: vk::CommandPool,
	image_type: vk::ImageType,
	format: vk::Format,
	extent: vk::Extent3D,
	mipmaps: bool,
	usage: vk::ImageUsageFlags,
	src: vk::Buffer,
) -> (vk::Image, Allocation, vk::ImageView) {
	unsafe {
		let mip_levels = if mipmaps { max_mipmaps(extent) } else { 1 };
		let ci = ash::vk::ImageCreateInfo::builder()
			.image_type(image_type)
			.format(format)
			.extent(extent)
			.mip_levels(mip_levels)
			.array_layers(1)
			.samples(vk::SampleCountFlags::TYPE_1)
			.usage(usage | vk::ImageUsageFlags::TRANSFER_DST);
		let aci = AllocationCreateInfo { usage: MemoryUsage::GpuOnly, ..Default::default() };
		let (image, allocation, _) = allocator.create_image(&ci, &aci).unwrap();

		let fence = device.create_fence(&vk::FenceCreateInfo::builder(), None).unwrap();

		let ci = vk::CommandBufferAllocateInfo::builder()
			.command_pool(cmdpool)
			.level(vk::CommandBufferLevel::PRIMARY)
			.command_buffer_count(1);
		let cmd = device.allocate_command_buffers(&ci).unwrap()[0];
		device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder()).unwrap();

		transition_layout(
			device,
			cmd,
			image,
			mip_levels,
			vk::ImageLayout::UNDEFINED,
			vk::ImageLayout::TRANSFER_DST_OPTIMAL,
			vk::PipelineStageFlags::empty(),
			vk::PipelineStageFlags::TRANSFER,
		);

		let copy = vk::BufferImageCopy::builder()
			.image_subresource(
				vk::ImageSubresourceLayers::builder().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1).build(),
			)
			.image_extent(extent)
			.build();
		device.cmd_copy_buffer_to_image(cmd, src, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[copy]);

		transition_layout(
			device,
			cmd,
			image,
			mip_levels,
			vk::ImageLayout::TRANSFER_DST_OPTIMAL,
			vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
			vk::PipelineStageFlags::TRANSFER,
			vk::PipelineStageFlags::FRAGMENT_SHADER,
		);

		device.end_command_buffer(cmd).unwrap();

		let submits = [vk::SubmitInfo::builder().command_buffers(&[cmd]).build()];
		device.queue_submit(queue, &submits, fence).unwrap();

		let view_type = match image_type {
			vk::ImageType::TYPE_1D => vk::ImageViewType::TYPE_1D,
			vk::ImageType::TYPE_2D => vk::ImageViewType::TYPE_2D,
			vk::ImageType::TYPE_3D => vk::ImageViewType::TYPE_3D,
			_ => unreachable!(),
		};
		let ci = vk::ImageViewCreateInfo::builder().image(image).view_type(view_type).format(format).subresource_range(
			vk::ImageSubresourceRange::builder()
				.aspect_mask(vk::ImageAspectFlags::COLOR)
				.level_count(mip_levels)
				.layer_count(1)
				.build(),
		);
		let image_view = device.create_image_view(&ci, None).unwrap();

		device.wait_for_fences(&[fence], false, !0).unwrap();

		device.destroy_fence(fence, None);
		device.free_command_buffers(cmdpool, &[cmd]);

		(image, allocation, image_view)
	}
}

pub(super) unsafe fn transition_layout(
	device: &Device,
	cmd: vk::CommandBuffer,
	image: vk::Image,
	level_count: u32,
	old_layout: vk::ImageLayout,
	new_layout: vk::ImageLayout,
	src_stage_mask: vk::PipelineStageFlags,
	dst_stage_mask: vk::PipelineStageFlags,
) {
	let src_access_mask = match old_layout {
		vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => vk::AccessFlags::SHADER_READ,
		vk::ImageLayout::TRANSFER_DST_OPTIMAL => vk::AccessFlags::TRANSFER_WRITE,
		vk::ImageLayout::UNDEFINED => vk::AccessFlags::empty(),
		_ => unimplemented!(),
	};
	let dst_access_mask = match new_layout {
		vk::ImageLayout::TRANSFER_DST_OPTIMAL => vk::AccessFlags::TRANSFER_WRITE,
		vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => vk::AccessFlags::SHADER_READ,
		_ => unimplemented!(),
	};

	let barrier = vk::ImageMemoryBarrier::builder()
		.src_access_mask(src_access_mask)
		.dst_access_mask(dst_access_mask)
		.old_layout(old_layout)
		.new_layout(new_layout)
		.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.image(image)
		.subresource_range(
			vk::ImageSubresourceRange::builder()
				.aspect_mask(vk::ImageAspectFlags::COLOR)
				.level_count(level_count)
				.layer_count(1)
				.build(),
		)
		.build();
	device
		.cmd_pipeline_barrier(cmd, src_stage_mask, dst_stage_mask, vk::DependencyFlags::empty(), &[], &[], &[barrier]);
}

fn max_mipmaps(extent: vk::Extent3D) -> u32 {
	32 - (extent.width | extent.height | extent.depth).leading_zeros()
}
