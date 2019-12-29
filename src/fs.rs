use crate::threads::FILE_THREAD;
use byteorder::{NativeEndian, ReadBytesExt};
use futures::{future::RemoteHandle, task::SpawnExt};
use std::{fs::File, io, mem::size_of, path::Path};

pub fn read_all_u32<P: AsRef<Path> + Send + 'static>(path: P) -> RemoteHandle<io::Result<Vec<u32>>> {
	FILE_THREAD
		.lock()
		.unwrap()
		.spawn_with_handle(async move {
			let mut file = File::open(path)?;
			let len = file.metadata()?.len() as usize;
			assert!(len % 4 == 0);
			let mut source = Vec::with_capacity(len / size_of::<u32>());
			loop {
				match file.read_u32::<NativeEndian>() {
					Ok(n) => source.push(n),
					Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => break,
					Err(err) => return Err(err),
				}
			}
			Ok::<_, io::Error>(source)
		})
		.unwrap()
}
