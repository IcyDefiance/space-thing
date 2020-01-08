use nalgebra::{partial_max, Scalar, Vector3};

pub fn lerp(lhs: f32, rhs: f32, t: f32) -> f32 {
	lhs + (rhs - lhs) * t
}

pub fn v3max<N: Scalar + PartialOrd>(lhs: Vector3<N>, rhs: N) -> Vector3<N> {
	[*partial_max(&lhs.x, &rhs).unwrap(), *partial_max(&lhs.y, &rhs).unwrap(), *partial_max(&lhs.z, &rhs).unwrap()]
		.into()
}
