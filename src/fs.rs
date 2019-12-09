use crate::threads::FILE_THREAD;
use futures::{future::RemoteHandle, task::SpawnExt};
use std::{
	fs::File,
	io::{self, prelude::*},
	path::Path,
};

pub fn read_bytes<P: AsRef<Path> + Send + 'static>(path: P) -> RemoteHandle<Result<Vec<u8>, io::Error>> {
	FILE_THREAD
		.lock()
		.unwrap()
		.spawn_with_handle(async move {
			let mut file = File::open(path)?;
			let mut source = vec![];
			file.read_to_end(&mut source)?;
			Ok::<_, io::Error>(source)
		})
		.unwrap()
}
