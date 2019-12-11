pub use ash::vk::BufferUsageFlags;
use std::marker::PhantomData;

use crate::gfx::Gfx;
use ash::{version::DeviceV1_0, vk};
use std::{mem::size_of, slice, sync::Arc};

pub struct ImmutableBuffer<T: ?Sized> {
	gfx: Arc<Gfx>,
	pub(super) buf: vk::Buffer,
	mem: vk::DeviceMemory,
	_phantom: PhantomData<Box<T>>,
}
impl<T: Clone> ImmutableBuffer<[T]> {
	pub fn from_slice(gfx: &Arc<Gfx>, data: &[T], usage: BufferUsageFlags) -> Self {
		let size = size_of::<T>() as u64 * data.len() as u64;
		let ci = vk::BufferCreateInfo::builder().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE);
		let buf = unsafe { gfx.device.create_buffer(&ci, None) }.unwrap();

		let reqs = unsafe { gfx.device.get_buffer_memory_requirements(buf) };
		let memflags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
		let memtype = find_memory_type(&gfx.memory_properties, reqs.memory_type_bits, memflags);
		let ci = vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(memtype);
		let mem = unsafe { gfx.device.allocate_memory(&ci, None) }.unwrap();

		unsafe { gfx.device.bind_buffer_memory(buf, mem, 0) }.unwrap();

		let bufdata = unsafe { gfx.device.map_memory(mem, 0, size, vk::MemoryMapFlags::empty()) }.unwrap();
		let bufdata = unsafe { slice::from_raw_parts_mut(bufdata as *mut T, (size / size_of::<T>() as u64) as _) };
		bufdata.clone_from_slice(data);
		unsafe { gfx.device.unmap_memory(mem) };

		Self { gfx: gfx.clone(), buf, mem, _phantom: PhantomData }
	}
}
impl<T: ?Sized> Drop for ImmutableBuffer<T> {
	fn drop(&mut self) {
		unsafe { self.gfx.device.destroy_buffer(self.buf, None) };
		unsafe { self.gfx.device.free_memory(self.mem, None) };
	}
}

fn find_memory_type(
	memprops: &vk::PhysicalDeviceMemoryProperties,
	type_filter: u32,
	properties: vk::MemoryPropertyFlags,
) -> u32 {
	(0..memprops.memory_type_count)
		.filter(|&i| {
			(type_filter & (1 << i) > 0) && !(memprops.memory_types[i as usize].property_flags & properties).is_empty()
		})
		.next()
		.unwrap()
}
