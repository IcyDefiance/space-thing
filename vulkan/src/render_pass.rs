use crate::device::Device;
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

pub struct RenderPass {
	device: Arc<Device>,
	pub vk: vk::RenderPass,
}
impl RenderPass {
	pub fn device(&self) -> &Arc<Device> {
		&self.device
	}

	pub unsafe fn from_vk(device: Arc<Device>, vk: vk::RenderPass) -> Arc<Self> {
		Arc::new(Self { device, vk })
	}
}
impl Drop for RenderPass {
	fn drop(&mut self) {
		unsafe { self.device.vk.destroy_render_pass(self.vk, None) };
	}
}

#[macro_export]
#[rustfmt::skip]
macro_rules! ordered_passes_renderpass {
	(
		$device:expr,
		attachments: {
			$(
				$atch_name:ident : {
					load: $load:ident,
					store: $store:ident,
					format: $format:expr,
					samples: $samples:expr,
					$(initial_layout: $init_layout:expr,)*
					$(final_layout: $final_layout:expr,)*
				}
			),*
		},
		passes: [
			$(
				{
					color: [$($color_atch:ident),*],
					depth_stencil: { $($depth_atch:ident)* },
					input: [$($input_atch:ident),*] $(,)*
					$(resolve: [$($resolve_atch:ident),*])* $(,)*
				}
			),*
		]
	) => {{
		let attachments = [$(
			vk::AttachmentDescription::builder()
				.format($format)
				.samples(vk::SampleCountFlags::TYPE_1)
				.load_op(vk::AttachmentLoadOp::CLEAR)
				.store_op(vk::AttachmentStoreOp::STORE)
				.stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
				.stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
				.initial_layout(vk::ImageLayout::UNDEFINED)
				.final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
				.build()
		),*];
		let color_attachments =
			[vk::AttachmentReference::builder().layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL).build()];
		let subpasses = [vk::SubpassDescription::builder()
			.pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
			.color_attachments(&color_attachments)
			.build()];
		let dependencies = [vk::SubpassDependency::builder()
			.src_subpass(vk::SUBPASS_EXTERNAL)
			.src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
			.dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
			.dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
			.build()];
		let ci = vk::RenderPassCreateInfo::builder()
			.attachments(&attachments)
			.subpasses(&subpasses)
			.dependencies(&dependencies);
		let vk = unsafe { $device.vk.create_render_pass(&ci, None) }.unwrap();
		unsafe { vulkan::render_pass::RenderPass::from_vk($device.clone(), vk) }
	}};
}
