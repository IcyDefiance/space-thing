use crate::gfx::{
	buffer::{BufferUsageFlags, ImmutableBuffer},
	window::Vertex,
	Gfx,
};
use std::sync::Arc;

pub struct Volume {
	pub(super) vertices: ImmutableBuffer<[Vertex]>,
	pub(super) indices: ImmutableBuffer<[u32]>,
}
impl Volume {
	pub async fn new(gfx: Arc<Gfx>, vertices: &[Vertex], indices: &[u32]) -> Arc<Self> {
		let vertices = ImmutableBuffer::from_slice(gfx.clone(), vertices, BufferUsageFlags::VERTEX_BUFFER);
		let indices = ImmutableBuffer::from_slice(gfx.clone(), indices, BufferUsageFlags::INDEX_BUFFER);

		Arc::new(Self { vertices, indices })
	}
}
