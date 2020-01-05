use nalgebra::{one, zero, UnitQuaternion, Vector3};

#[repr(C)]
pub struct Camera {
	pub pos: Vector3<f32>,
	dummy: f32,
	pub rot: UnitQuaternion<f32>,
	pub yaw: f32,
	pub pitch: f32,
	pub sensitivity: f32
}
impl Camera {
	pub fn new() -> Self {
		Self { pos: zero(), dummy: 0.0, rot: one(), yaw: 0.0, pitch: 0.0, sensitivity: 1.0 }
	}
	pub fn look(&mut self, x: f32, y: f32) {
		self.yaw -= x * self.sensitivity;
		self.pitch -= y * self.sensitivity;
		self.update();
	}
	pub fn update(&mut self) {
		self.rot = UnitQuaternion::from_euler_angles(0.0, 0.0, self.yaw) * UnitQuaternion::from_euler_angles(self.pitch, 0.0, 0.0);
	}
}
