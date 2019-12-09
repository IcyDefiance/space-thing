use futures::{executor::ThreadPool, future::lazy, task::SpawnExt};
use lazy_static::lazy_static;
use std::{
	future::Future,
	pin::Pin,
	sync::Mutex,
	task::{Context, Poll},
};

lazy_static! {
	pub static ref FILE_THREAD: Mutex<ThreadPool> = Mutex::new(ThreadPool::builder().pool_size(1).create().unwrap());
	pub static ref WAKER_THREAD: Mutex<ThreadPool> = Mutex::new(ThreadPool::builder().pool_size(1).create().unwrap());
}

pub fn yield_once() -> YieldOnce {
	YieldOnce { yielded: false }
}

pub struct YieldOnce {
	yielded: bool,
}
impl Future for YieldOnce {
	type Output = ();

	fn poll(mut self: Pin<&mut Self>, cx: &mut Context) -> Poll<Self::Output> {
		if self.yielded {
			Poll::Ready(())
		} else {
			self.yielded = true;
			let waker = cx.waker().clone();
			WAKER_THREAD.lock().unwrap().spawn(lazy(move |_| waker.wake())).unwrap();
			Poll::Pending
		}
	}
}
