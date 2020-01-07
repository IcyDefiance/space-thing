pub fn lerp(lhs: f32, rhs: f32, t: f32) -> f32 {
	lhs + (rhs - lhs) * t
}
