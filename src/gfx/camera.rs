use nalgebra::{one, zero, UnitQuaternion, Vector3};

#[repr(C)]
pub struct Camera {
	pub pos: Vector3<f32>,
	dummy: f32,
	pub rot: UnitQuaternion<f32>,
}
impl Camera {
	pub fn new() -> Self {
		Self { pos: zero(), dummy: 0.0, rot: one() }
	}
}
