use crate::gfx::{
	buffer::{BufferUsageFlags, ImmutableBuffer},
	window::Vertex,
	Gfx,
};
use std::sync::Arc;

pub struct Volume {
	vertices: ImmutableBuffer<[Vertex]>,
	indices: ImmutableBuffer<[u32]>,
}
impl Volume {
	pub async fn new(gfx: Arc<Gfx>, vertices: &[Vertex], indices: &[u32]) -> Self {
		let vertices = ImmutableBuffer::from_slice(&gfx, vertices, BufferUsageFlags::VERTEX_BUFFER);
		let indices = ImmutableBuffer::from_slice(&gfx, indices, BufferUsageFlags::INDEX_BUFFER);
		let vertices = vertices.await;
		let indices = indices.await;
		Self { vertices, indices }
	}
}
