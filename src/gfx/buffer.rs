use crate::gfx::vulkan::Fence;
pub use ash::vk::BufferUsageFlags;
use std::{marker::PhantomData, u64};

use crate::gfx::Gfx;
use ash::{version::DeviceV1_0, vk};
use std::{mem::size_of, slice, sync::Arc};

pub struct ImmutableBuffer<T: ?Sized> {
	pub(super) gfx: Arc<Gfx>,
	pub(super) buf: vk::Buffer,
	mem: vk::DeviceMemory,
	pub(super) len: usize,
	_phantom: PhantomData<Box<T>>,
}
impl<T: Clone> ImmutableBuffer<[T]> {
	pub async fn from_slice(gfx: Arc<Gfx>, data: &[T], usage: BufferUsageFlags) -> Self {
		unsafe {
			let len = data.len();
			let size = size_of::<T>() as u64 * len as u64;

			let (cpubuf, cpumem) = create_buffer(
				&gfx,
				size,
				BufferUsageFlags::TRANSFER_SRC,
				vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
			);

			let bufdata = gfx.device.map_memory(cpumem, 0, size, vk::MemoryMapFlags::empty()).unwrap();
			let bufdata = slice::from_raw_parts_mut(bufdata as *mut T, (size / size_of::<T>() as u64) as _);
			bufdata.clone_from_slice(data);
			gfx.device.unmap_memory(cpumem);

			let (buf, mem) = create_buffer(
				&gfx,
				size,
				BufferUsageFlags::TRANSFER_DST | usage,
				vk::MemoryPropertyFlags::DEVICE_LOCAL,
			);

			let fence = Fence::new(gfx.clone(), false);

			let ci = vk::CommandBufferAllocateInfo::builder()
				.command_pool(gfx.cmdpool_transient)
				.level(vk::CommandBufferLevel::PRIMARY)
				.command_buffer_count(1);
			let cmds = gfx.device.allocate_command_buffers(&ci).unwrap();
			let cmd = cmds[0];
			gfx.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder()).unwrap();
			gfx.device.cmd_copy_buffer(cmd, cpubuf, buf, &[vk::BufferCopy::builder().size(size).build()]);
			gfx.device.end_command_buffer(cmd).unwrap();

			let submits = [vk::SubmitInfo::builder().command_buffers(&cmds).build()];
			gfx.device.queue_submit(gfx.queue, &submits, fence.vk).unwrap();

			fence.await.unwrap();

			gfx.device.free_command_buffers(gfx.cmdpool_transient, &cmds);
			gfx.device.destroy_buffer(cpubuf, None);
			gfx.device.free_memory(cpumem, None);

			Self { gfx, buf, mem, len, _phantom: PhantomData }
		}
	}
}
impl<T: ?Sized> Drop for ImmutableBuffer<T> {
	fn drop(&mut self) {
		unsafe {
			self.gfx.device.destroy_buffer(self.buf, None);
			self.gfx.device.free_memory(self.mem, None);
		}
	}
}

fn create_buffer(
	gfx: &Gfx,
	size: u64,
	usage: BufferUsageFlags,
	memflags: vk::MemoryPropertyFlags,
) -> (vk::Buffer, vk::DeviceMemory) {
	let ci = vk::BufferCreateInfo::builder().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE);
	let buf = unsafe { gfx.device.create_buffer(&ci, None) }.unwrap();

	let reqs = unsafe { gfx.device.get_buffer_memory_requirements(buf) };
	let memtype = find_memory_type(&gfx.memory_properties, reqs.memory_type_bits, memflags);
	let ci = vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(memtype);
	let mem = unsafe { gfx.device.allocate_memory(&ci, None) }.unwrap();

	unsafe { gfx.device.bind_buffer_memory(buf, mem, 0) }.unwrap();

	(buf, mem)
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
