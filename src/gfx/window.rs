use crate::gfx::Gfx;
use std::sync::Arc;
use winit::{EventsLoop, WindowBuilder};

pub struct Window {
	gfx: Arc<Gfx>,
	window: winit::Window,
}
impl Window {
	pub fn new(gfx: &Arc<Gfx>, events_loop: &EventsLoop) -> Self {
		let window = WindowBuilder::new().with_dimensions((1440, 810).into()).build(&events_loop).unwrap();
		Self { gfx: gfx.clone(), window }
	}
}
