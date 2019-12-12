use crate::gfx::{
	buffer::{BufferUsageFlags, ImmutableBuffer},
	window::{Vertex, Window, WindowInner},
	Gfx,
};
use ash::{version::DeviceV1_0, vk};
use std::sync::{Arc, Mutex};

pub struct VolumeData {
	vertices: ImmutableBuffer<[Vertex]>,
	indices: ImmutableBuffer<[u32]>,
}
impl VolumeData {
	pub async fn new(gfx: Arc<Gfx>, vertices: &[Vertex], indices: &[u32]) -> Arc<Self> {
		let vertices = ImmutableBuffer::from_slice(gfx.clone(), vertices, BufferUsageFlags::VERTEX_BUFFER);
		let indices = ImmutableBuffer::from_slice(gfx.clone(), indices, BufferUsageFlags::INDEX_BUFFER);
		let vertices = vertices.await;
		let indices = indices.await;

		Arc::new(Self { vertices, indices })
	}
}

pub struct StaticVolume {
	window: Arc<Window>,
	data: Arc<VolumeData>,
	cmds: Mutex<(vk::Pipeline, Vec<vk::CommandBuffer>)>,
}
impl StaticVolume {
	pub fn new(window: Arc<Window>, data: Arc<VolumeData>) -> Arc<Self> {
		Arc::new(Self { window, data, cmds: Mutex::default() })
	}

	pub(super) fn get_cmds(&self, window_inner: &WindowInner, image_idx: usize) -> vk::CommandBuffer {
		let mut cmds_lock = self.cmds.lock().unwrap();
		let (pipeline, cmds) = &mut *cmds_lock;

		if *pipeline != window_inner.pipeline {
			let gfx = &self.data.vertices.gfx;

			unsafe {
				gfx.device.free_command_buffers(gfx.cmdpool, &cmds);

				let ci = vk::CommandBufferAllocateInfo::builder()
					.command_pool(gfx.cmdpool)
					.level(vk::CommandBufferLevel::SECONDARY)
					.command_buffer_count(window_inner.framebuffers.len() as _);
				let newcmds = gfx.device.allocate_command_buffers(&ci).unwrap();

				for (&cmd, &fb) in newcmds.iter().zip(&window_inner.framebuffers) {
					let inherit = vk::CommandBufferInheritanceInfo::builder()
						.render_pass(self.window.render_pass)
						.subpass(0)
						.framebuffer(fb);
					let info = vk::CommandBufferBeginInfo::builder()
						.flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE)
						.inheritance_info(&inherit);
					gfx.device.begin_command_buffer(cmd, &info).unwrap();
					gfx.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, window_inner.pipeline);
					gfx.device.cmd_bind_vertex_buffers(cmd, 0, &[self.data.vertices.buf], &[0]);
					gfx.device.cmd_bind_index_buffer(cmd, self.data.indices.buf, 0, vk::IndexType::UINT32);
					gfx.device.cmd_draw_indexed(cmd, self.data.indices.len as _, 1, 0, 0, 0);
					gfx.device.end_command_buffer(cmd).unwrap();
				}

				*pipeline = window_inner.pipeline;
				*cmds = newcmds;
			}
		}

		cmds[image_idx]
	}
}
impl Drop for StaticVolume {
	fn drop(&mut self) {
		let gfx = &self.data.vertices.gfx;
		let cmds_lock = self.cmds.lock().unwrap();
		let (_, cmds) = &*cmds_lock;
		unsafe { gfx.device.free_command_buffers(gfx.cmdpool, cmds) };
	}
}
