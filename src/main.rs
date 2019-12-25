mod fs;
mod gfx;
mod threads;

use futures::executor::block_on;
use gfx::{window::Window, Gfx};
use simplelog::{LevelFilter, SimpleLogger};
use winit::{
	event::{Event, KeyboardInput, VirtualKeyCode, WindowEvent},
	event_loop::{ControlFlow, EventLoop},
};

fn main() {
	block_on(amain());
}

async fn amain() {
	SimpleLogger::init(LevelFilter::Warn, Default::default()).unwrap();

	let gfx = Gfx::new().await;

	let event_loop = EventLoop::new();
	let mut window = Window::new(gfx.clone(), &event_loop);

	event_loop.run(move |event, _window, control| {
		*control = ControlFlow::Poll;

		match event {
			Event::WindowEvent { event, .. } => match event {
				WindowEvent::CloseRequested => *control = ControlFlow::Exit,
				WindowEvent::KeyboardInput { input: KeyboardInput { virtual_keycode, .. }, .. } => {
					match virtual_keycode {
						Some(VirtualKeyCode::Escape) => *control = ControlFlow::Exit,
						_ => (),
					}
				},
				_ => (),
			},
			Event::EventsCleared => window.draw(),
			_ => (),
		};
	});
}
