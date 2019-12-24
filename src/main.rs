mod fs;
mod gfx;
mod threads;

use crate::gfx::window::Vertex;
use futures::executor::block_on;
use gfx::{volume::Volume, window::Window, Gfx};
use simplelog::{LevelFilter, SimpleLogger};
use winit::{Event, EventsLoop, WindowEvent};

fn main() {
	block_on(amain());
}

async fn amain() {
	SimpleLogger::init(LevelFilter::Warn, Default::default()).unwrap();

	let gfx = Gfx::new().await;

	let mut events_loop = EventsLoop::new();
	let mut window = Window::new(gfx.clone(), &events_loop);

	let data1 = Volume::new(
		gfx.clone(),
		&[
			Vertex { pos: [-0.25, -0.25].into(), color: [1.0, 0.0, 0.0].into() },
			Vertex { pos: [0.75, -0.25].into(), color: [0.0, 1.0, 0.0].into() },
			Vertex { pos: [0.75, 0.75].into(), color: [0.0, 0.0, 1.0].into() },
			Vertex { pos: [-0.25, 0.75].into(), color: [1.0, 1.0, 1.0].into() },
		],
		&[0u32, 1, 2, 2, 3, 0],
	)
	.await;
	let data2 = Volume::new(
		gfx.clone(),
		&[
			Vertex { pos: [-0.75, -0.75].into(), color: [1.0, 0.0, 0.0].into() },
			Vertex { pos: [0.25, -0.75].into(), color: [0.0, 1.0, 0.0].into() },
			Vertex { pos: [0.25, 0.25].into(), color: [0.0, 0.0, 1.0].into() },
			Vertex { pos: [-0.75, 0.25].into(), color: [1.0, 1.0, 1.0].into() },
		],
		&[0u32, 1, 2, 2, 3, 0],
	)
	.await;
	let volumes = vec![data1, data2];

	loop {
		let mut exit = false;
		events_loop.poll_events(|event| match event {
			Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => exit = true,
			_ => (),
		});
		if exit {
			break;
		}

		window.draw(volumes.clone());
	}
}
