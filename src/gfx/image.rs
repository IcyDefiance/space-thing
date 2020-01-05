use ash::{version::DeviceV1_0, vk, Device};
use std::{mem::size_of, slice, u64};
use vk_mem::{Allocation, AllocationCreateInfo, Allocator, MemoryUsage};

pub(super) fn create_device_local_image(
	device: &Device,
	queue: vk::Queue,
	allocator: &Allocator,
	cmdpool: vk::CommandPool,
	image_type: vk::ImageType,
	format: vk::Format,
	extent: vk::Extent3D,
	usage: vk::ImageUsageFlags,
	size: u64,
	cb: impl FnOnce(&mut [u8]),
) -> (vk::Image, Allocation, vk::ImageView) {
	unsafe {
		let ci = ash::vk::BufferCreateInfo::builder().size(size).usage(vk::BufferUsageFlags::TRANSFER_SRC);
		let aci = AllocationCreateInfo { usage: MemoryUsage::CpuOnly, ..Default::default() };
		let (cpubuf, cpualloc, _) = allocator.create_buffer(&ci, &aci).unwrap();

		let bufdata = allocator.map_memory(&cpualloc).unwrap();
		let mut bufdata = slice::from_raw_parts_mut(bufdata, size as _);
		cb(&mut bufdata);
		allocator.unmap_memory(&cpualloc).unwrap();

		let ci = ash::vk::ImageCreateInfo::builder()
			.image_type(vk::ImageType::TYPE_3D)
			.image_type(image_type)
			.format(format)
			.extent(extent)
			.mip_levels(1)
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

		transition_layout(device, cmd, image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);

		let copy = vk::BufferImageCopy::builder()
			.image_subresource(
				vk::ImageSubresourceLayers::builder().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1).build(),
			)
			.image_extent(extent)
			.build();
		device.cmd_copy_buffer_to_image(cmd, cpubuf, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[copy]);

		transition_layout(
			device,
			cmd,
			image,
			vk::ImageLayout::TRANSFER_DST_OPTIMAL,
			vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
		);

		device.end_command_buffer(cmd).unwrap();

		let submits = [vk::SubmitInfo::builder().command_buffers(&[cmd]).build()];
		device.queue_submit(queue, &submits, fence).unwrap();

		let ci = vk::ImageViewCreateInfo::builder()
			.image(image)
			.view_type(vk::ImageViewType::TYPE_3D)
			.format(format)
			.subresource_range(
				vk::ImageSubresourceRange::builder()
					.aspect_mask(vk::ImageAspectFlags::COLOR)
					.level_count(1)
					.layer_count(1)
					.build(),
			);
		let image_view = device.create_image_view(&ci, None).unwrap();

		device.wait_for_fences(&[fence], false, !0).unwrap();

		device.destroy_fence(fence, None);
		device.free_command_buffers(cmdpool, &[cmd]);
		device.destroy_buffer(cpubuf, None);
		allocator.free_memory(&cpualloc).unwrap();

		(image, allocation, image_view)
	}
}

unsafe fn transition_layout(
	device: &Device,
	cmd: vk::CommandBuffer,
	image: vk::Image,
	old_layout: vk::ImageLayout,
	new_layout: vk::ImageLayout,
) {
	let (src_access_mask, src_stage_mask) = match old_layout {
		vk::ImageLayout::UNDEFINED => (vk::AccessFlags::empty(), vk::PipelineStageFlags::TOP_OF_PIPE),
		vk::ImageLayout::TRANSFER_DST_OPTIMAL => (vk::AccessFlags::TRANSFER_WRITE, vk::PipelineStageFlags::TRANSFER),
		_ => unimplemented!(),
	};
	let (dst_access_mask, dst_stage_mask) = match new_layout {
		vk::ImageLayout::TRANSFER_DST_OPTIMAL => (vk::AccessFlags::TRANSFER_WRITE, vk::PipelineStageFlags::TRANSFER),
		vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => {
			(vk::AccessFlags::SHADER_READ, vk::PipelineStageFlags::FRAGMENT_SHADER)
		},
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
				.level_count(1)
				.layer_count(1)
				.build(),
		)
		.build();
	device
		.cmd_pipeline_barrier(cmd, src_stage_mask, dst_stage_mask, vk::DependencyFlags::empty(), &[], &[], &[barrier]);
}
