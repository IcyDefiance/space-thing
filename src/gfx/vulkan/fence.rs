use crate::{gfx::Gfx, threads::WAKER_THREAD};
use ash::{version::DeviceV1_0, vk};
use futures::task::SpawnExt;
use std::{
	future::Future,
	pin::Pin,
	sync::Arc,
	task::{Context, Poll},
	u64,
};

pub struct Fence {
	gfx: Arc<Gfx>,
	pub(in super::super) vk: vk::Fence,
}
impl Fence {
	pub fn new(gfx: &Arc<Gfx>, signalled: bool) -> Self {
		// TODO: maybe use fence pool
		let flags = if signalled { vk::FenceCreateFlags::SIGNALED } else { vk::FenceCreateFlags::empty() };
		let vk = unsafe { gfx.device.create_fence(&vk::FenceCreateInfo::builder().flags(flags), None) }.unwrap();
		Self { gfx: gfx.clone(), vk }
	}

	pub fn reset(&self) {
		unsafe { self.gfx.device.reset_fences(&[self.vk]) }.unwrap();
	}

	pub fn wait(&self, timeout: u64) {
		unsafe { self.gfx.device.wait_for_fences(&[self.vk], true, timeout) }.unwrap();
	}
}
impl Future for Fence {
	type Output = Result<(), vk::Result>;

	fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
		match unsafe { self.gfx.device.get_fence_status(self.vk) } {
			Ok(()) => Poll::Ready(Ok(())),
			Err(vk::Result::NOT_READY) => {
				let waker = cx.waker().clone();
				WAKER_THREAD.lock().unwrap().spawn(async move { waker.wake() }).unwrap();
				Poll::Pending
			},
			Err(err) => Poll::Ready(Err(err)),
		}
	}
}
impl Drop for Fence {
	fn drop(&mut self) {
		self.wait(u64::MAX);
		unsafe { self.gfx.device.destroy_fence(self.vk, None) };
	}
}
