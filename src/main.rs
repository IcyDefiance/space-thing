mod fs;
mod gfx;
mod threads;

use futures::executor::block_on;
use gfx::{
	buffer::{BufferUsageFlags, ImmutableBuffer},
	window::{Vertex, Window},
	Gfx,
};
use simplelog::{LevelFilter, SimpleLogger};
use winit::{Event, EventsLoop, WindowEvent};

fn main() {
	block_on(amain());
}

async fn amain() {
	SimpleLogger::init(LevelFilter::Warn, Default::default()).unwrap();

	let gfx = Gfx::new().await;

	let mut events_loop = EventsLoop::new();
	let mut window = Window::new(&gfx, &events_loop);

	loop {
		let mut exit = false;
		events_loop.poll_events(|event| match event {
			Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => exit = true,
			_ => (),
		});
		if exit {
			break;
		}

		if window.draw() {
			window.recreate_swapchain();
		}
	}
}
