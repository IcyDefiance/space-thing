mod fs;
mod gfx;
mod threads;

use futures::executor::block_on;
use gfx::{camera::Camera, window::Window, world::World, Gfx};
use nalgebra::{zero, UnitQuaternion, Vector2, Vector3};
use simplelog::{LevelFilter, SimpleLogger};
use std::{collections::HashSet, time::Instant};
use winit::{
	event::{DeviceEvent, ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
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

	let world = World::new(gfx);
	let mut camera = Camera::new();
	camera.pos = [0.0, -4.0, 1.618].into();
	println!("{:?}", camera.rot);

	let mut keys = HashSet::new();
	let mut rotation: Vector2<f32> = [0.0, 0.0].into();

	let mut time = Instant::now();
	let mut delta = time.elapsed();
	event_loop.run(move |event, _window, control| {
		*control = ControlFlow::Poll;

		match event {
			Event::DeviceEvent { event, .. } => match event {
				DeviceEvent::Key(KeyboardInput { virtual_keycode, state, .. }) => match virtual_keycode {
					Some(virtual_keycode) => match state {
						ElementState::Pressed => {
							keys.insert(virtual_keycode);
						},
						ElementState::Released => {
							keys.remove(&virtual_keycode);
						},
					},
					None => (),
				},
				DeviceEvent::MouseMotion { delta } => rotation += Vector2::from([delta.0 as _, delta.1 as _]),
				_ => (),
			},
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
			Event::EventsCleared => {
				let now = Instant::now();
				delta = now - time;
				time = now;

				let speed = 10.0;
				let mut movement = Vector3::from([
					(keys.contains(&VirtualKeyCode::D) as i32 - keys.contains(&VirtualKeyCode::A) as i32) as f32,
					(keys.contains(&VirtualKeyCode::W) as i32 - keys.contains(&VirtualKeyCode::S) as i32) as f32,
					(keys.contains(&VirtualKeyCode::Space) as i32 - keys.contains(&VirtualKeyCode::LShift) as i32) as f32,
				]);
				movement *= delta.as_secs_f32() * speed;

				let mouse_sensitivity = 0.005;
				rotation *= mouse_sensitivity;

				camera.look(rotation.x, rotation.y);
				camera.pos += camera.rot * movement;

				rotation = zero();

				window.draw(&world, &camera);
			},
			_ => (),
		};
	});
}
