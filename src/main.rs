mod fs;
mod gfx;
mod threads;

use futures::executor::block_on;
use gfx::{window::Window, Gfx};
use simplelog::{LevelFilter, SimpleLogger};
use winit::{ControlFlow, Event, EventsLoop, WindowEvent};

fn main() {
	block_on(amain());
}

async fn amain() {
	SimpleLogger::init(LevelFilter::Warn, Default::default()).unwrap();

	let gfx = Gfx::new().await;

	let mut events_loop = EventsLoop::new();
	let window = Window::new(&gfx, &events_loop);

	events_loop.run_forever(|event| match event {
		Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => ControlFlow::Break,
		_ => ControlFlow::Continue,
	});
}
