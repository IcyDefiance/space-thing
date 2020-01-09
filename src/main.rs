mod fs;
mod gfx;
mod threads;

use futures::executor::block_on;
use gfx::{camera::Camera, window::Window, world::World, Gfx};
use nalgebra::{zero, Vector2, Vector3};
use simplelog::{LevelFilter, SimpleLogger};
use std::{collections::HashSet, time::Instant};
use winit::{
	event::{DeviceEvent, ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent},
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
	grab_cursor(&window, true);

	let mut world = World::new(gfx);
	world.set_block([0, 0, 5].into());

	let mut camera = Camera::new();
	camera.pos = [8.0, 8.0, 8.0].into();
	camera.yaw = 3.14159;

	let mut keys = HashSet::new();
	let mut rotation: Vector2<f32> = [0.0, 0.0].into();
	let mut controls = true;

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
				WindowEvent::Focused(false) => {
					controls = false;
					grab_cursor(&window, false);
				},
				WindowEvent::KeyboardInput {
					input: KeyboardInput { virtual_keycode, state: ElementState::Pressed, .. },
					..
				} => match virtual_keycode {
					Some(VirtualKeyCode::Escape) => {
						if controls {
							controls = false;
							grab_cursor(&window, false);
						} else {
							*control = ControlFlow::Exit;
						}
					},
					_ => (),
				},
				WindowEvent::MouseInput { button: MouseButton::Left, .. } => {
					controls = true;
					grab_cursor(&window, true);
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
					(keys.contains(&VirtualKeyCode::Space) as i32 - keys.contains(&VirtualKeyCode::LShift) as i32)
						as f32,
				]);
				movement *= delta.as_secs_f32() * speed;

				let mouse_sensitivity = 0.005;
				rotation *= mouse_sensitivity;

				if controls {
					camera.look(rotation.x, rotation.y);
					camera.pos += camera.rot * movement;
				}

				rotation = zero();

				window.draw(&mut world, &camera);
			},
			_ => (),
		};
	});
}

fn grab_cursor(window: &Window, grab: bool) {
	let winit = window.winit();
	winit.set_cursor_grab(grab).unwrap();
	winit.set_cursor_visible(!grab);
}
