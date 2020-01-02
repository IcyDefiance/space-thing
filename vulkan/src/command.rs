pub use ash::vk::ClearValue;

use crate::{
	buffer::{Buffer, BufferAbstract},
	device::Device,
	image::Framebuffer,
	pipeline::Pipeline,
	render_pass::RenderPass,
	sync::Resource,
	Rect2D,
};
use ash::{version::DeviceV1_0, vk};
use std::{
	cell::{RefCell, RefMut},
	collections::HashMap,
	marker::PhantomData,
	sync::{Arc, Mutex},
};
use thread_local::ThreadLocal;
use typenum::{Bit, B0, B1};

pub struct CommandPool {
	device: Arc<Device>,
	pub(crate) queue_family: u32,
	transient: bool,
	pools: ThreadLocal<RefCell<CommandPoolInner>>,
	free: Mutex<HashMap<vk::CommandPool, CmdCollection>>,
}
impl CommandPool {
	pub fn record(self: &Arc<CommandPool>, one_time: bool, simultaneous: bool) -> CommandBufferBuilder<B0> {
		let cmd = self.get_cmdbuf(false);
		unsafe {
			self.begin(cmd, one_time, simultaneous, &None);
			CommandBufferBuilder::from_vk(self.clone(), self.get_pool().vk, one_time, simultaneous, None, cmd)
		}
	}

	pub fn record_secondary(
		self: &Arc<CommandPool>,
		one_time: bool,
		simultaneous: bool,
		inherit: Option<InheritanceInfo>,
	) -> CommandBufferBuilder<B1> {
		let cmd = self.get_cmdbuf(true);
		unsafe {
			self.begin(cmd, one_time, simultaneous, &inherit);
			CommandBufferBuilder::from_vk(self.clone(), self.get_pool().vk, one_time, simultaneous, inherit, cmd)
		}
	}

	/// Resets the thread-local pool.
	///
	/// Panics if any command buffers have been built from the thread-local pool but not dropped.
	pub fn reset(&self, release: bool) {
		let mut pool = self.get_pool();

		let (free_primary, free_secondary) = {
			let free_lock = self.free.lock().unwrap();
			let free = free_lock.get(&pool.vk).unwrap();
			(free.primary.len(), free.secondary.len())
		};
		assert!(pool.cmds.primary.len() + free_primary == pool.primary_size as _);
		assert!(pool.cmds.secondary.len() + free_secondary == pool.secondary_size as _);

		let flags =
			if release { vk::CommandPoolResetFlags::RELEASE_RESOURCES } else { vk::CommandPoolResetFlags::empty() };
		unsafe { self.device.vk.reset_command_pool(pool.vk, flags) }.unwrap();

		let mut free_lock = self.free.lock().unwrap();
		let free = free_lock.get_mut(&pool.vk).unwrap();
		pool.cmds.primary.extend(free.primary.drain(..));
		pool.cmds.secondary.extend(free.secondary.drain(..));
	}

	pub(crate) unsafe fn from_vk(device: Arc<Device>, queue_family: u32, transient: bool) -> Arc<Self> {
		Arc::new(Self { device, queue_family, transient, pools: ThreadLocal::new(), free: Mutex::default() })
	}

	unsafe fn begin(
		&self,
		cmd: vk::CommandBuffer,
		one_time: bool,
		simultaneous: bool,
		inherit: &Option<InheritanceInfo>,
	) {
		let mut flags = vk::CommandBufferUsageFlags::empty();
		if one_time {
			flags |= vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT;
		}
		if simultaneous {
			flags |= vk::CommandBufferUsageFlags::SIMULTANEOUS_USE;
		}
		let mut inheritance_info = vk::CommandBufferInheritanceInfo::builder();
		if let Some(inherit) = inherit {
			flags |= vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE;
			inheritance_info = inheritance_info
				.render_pass(inherit.render_pass.vk)
				.subpass(inherit.subpass)
				.framebuffer(inherit.framebuffer.as_ref().map(|x| x.vk).unwrap_or(vk::Framebuffer::null()));
		}

		let bi = vk::CommandBufferBeginInfo::builder().flags(flags).inheritance_info(&inheritance_info);
		self.device.vk.begin_command_buffer(cmd, &bi).unwrap();
	}

	fn get_cmdbuf(&self, secondary: bool) -> vk::CommandBuffer {
		let mut pool = self.get_pool();

		let cmdslen = if secondary { pool.cmds.secondary.len() } else { pool.cmds.primary.len() };
		if cmdslen == 0 {
			let (level, mut size) = if secondary {
				(vk::CommandBufferLevel::SECONDARY, pool.secondary_size)
			} else {
				(vk::CommandBufferLevel::PRIMARY, pool.primary_size)
			};
			if size == 0 {
				size = 1;
			}

			let ci =
				vk::CommandBufferAllocateInfo::builder().command_pool(pool.vk).level(level).command_buffer_count(size);
			let cmds = unsafe { self.device.vk.allocate_command_buffers(&ci) }.unwrap();

			if secondary {
				pool.cmds.secondary = cmds;
				pool.secondary_size += size;
			} else {
				pool.cmds.primary = cmds;
				pool.primary_size += size;
			}
		}

		if secondary { pool.cmds.secondary.pop() } else { pool.cmds.primary.pop() }.unwrap()
	}

	fn get_pool(&self) -> RefMut<CommandPoolInner> {
		self.pools
			.get_or(|| {
				let mut flags = vk::CommandPoolCreateFlags::empty();
				if self.transient {
					flags |= vk::CommandPoolCreateFlags::TRANSIENT;
				};

				let ci = vk::CommandPoolCreateInfo::builder().flags(flags).queue_family_index(self.queue_family);
				let vk = unsafe { self.device.vk.create_command_pool(&ci, None) }.unwrap();

				self.free.lock().unwrap().insert(vk, CmdCollection::new());

				RefCell::new(CommandPoolInner { vk, cmds: CmdCollection::new(), primary_size: 0, secondary_size: 0 })
			})
			.borrow_mut()
	}
}
impl Drop for CommandPool {
	fn drop(&mut self) {
		for pool in self.pools.iter_mut() {
			unsafe { self.device.vk.destroy_command_pool(pool.get_mut().vk, None) };
		}
	}
}

struct CommandPoolInner {
	vk: vk::CommandPool,
	cmds: CmdCollection,
	primary_size: u32,
	secondary_size: u32,
}

struct CmdCollection {
	primary: Vec<vk::CommandBuffer>,
	secondary: Vec<vk::CommandBuffer>,
}
impl CmdCollection {
	fn new() -> Self {
		Self { primary: vec![], secondary: vec![] }
	}
}

pub struct CommandBufferBuilder<SEC: Bit> {
	pool: Arc<CommandPool>,
	vkpool: vk::CommandPool,
	_one_time: bool,
	_simultaneous: bool,
	_inherit: Option<InheritanceInfo>,
	vk: vk::CommandBuffer,
	resources: Vec<Resource>,
	sec: PhantomData<SEC>,
}
impl<SEC: Bit> CommandBufferBuilder<SEC> {
	unsafe fn from_vk(
		pool: Arc<CommandPool>,
		vkpool: vk::CommandPool,
		one_time: bool,
		simultaneous: bool,
		inherit: Option<InheritanceInfo>,
		vk: vk::CommandBuffer,
	) -> Self {
		Self {
			pool,
			vkpool,
			_one_time: one_time,
			_simultaneous: simultaneous,
			_inherit: inherit,
			vk,
			resources: vec![],
			sec: PhantomData,
		}
	}

	pub fn begin_render_pass(
		mut self,
		render_pass: Arc<RenderPass>,
		framebuffer: Arc<Framebuffer>,
		render_area: Rect2D,
		clear_values: &[ClearValue],
	) -> Self {
		let ci = vk::RenderPassBeginInfo::builder()
			.render_pass(render_pass.vk)
			.framebuffer(framebuffer.vk)
			.render_area(render_area)
			.clear_values(clear_values);
		unsafe {
			self.pool.device.vk.cmd_begin_render_pass(self.vk, &ci, vk::SubpassContents::SECONDARY_COMMAND_BUFFERS)
		};

		self.resources.push(Resource::RenderPass(render_pass));
		self.resources.push(Resource::Framebuffer(framebuffer));
		self
	}

	pub fn build(self) -> Arc<CommandBuffer<SEC>> {
		unsafe {
			self.pool.device.vk.end_command_buffer(self.vk).unwrap();
			CommandBuffer::from_vk(self.pool, self.vkpool, self.vk, self.resources)
		}
	}

	pub fn bind_pipeline(mut self, pipeline: Arc<Pipeline>) -> Self {
		unsafe { self.pool.device.vk.cmd_bind_pipeline(self.vk, vk::PipelineBindPoint::GRAPHICS, pipeline.vk) };
		self.resources.push(Resource::Pipeline(pipeline));
		self
	}

	pub fn bind_vertex_buffers(
		mut self,
		first_binding: u32,
		buffers: impl IntoIterator<Item = Arc<dyn BufferAbstract>>,
		offsets: &[u64],
	) -> Self {
		let buffers = buffers.into_iter();
		let (lower, upper) = buffers.size_hint();
		let mut buffer_vks = Vec::with_capacity(upper.unwrap_or(lower));
		for buf in buffers {
			buffer_vks.push(buf.vk());
			self.resources.push(Resource::Buffer(buf));
		}

		unsafe { self.pool.device.vk.cmd_bind_vertex_buffers(self.vk, first_binding, &buffer_vks, offsets) };
		self
	}

	pub fn copy_buffer<T: ?Sized + 'static>(mut self, src: Arc<Buffer<T>>, dst: Arc<Buffer<T>>) -> Self {
		assert!(src.size() <= dst.size());

		let regions = [vk::BufferCopy::builder().size(src.size()).build()];
		unsafe { self.pool.device.vk.cmd_copy_buffer(self.vk, src.vk, dst.vk, &regions) };

		self.resources.push(Resource::Buffer(src));
		self.resources.push(Resource::Buffer(dst));
		self
	}

	pub fn draw(self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) -> Self {
		unsafe { self.pool.device.vk.cmd_draw(self.vk, vertex_count, instance_count, first_vertex, first_instance) };
		self
	}

	pub fn end_render_pass(self) -> Self {
		unsafe { self.pool.device.vk.cmd_end_render_pass(self.vk) };
		self
	}

	pub fn execute_commands(mut self, secondaries: impl IntoIterator<Item = Arc<CommandBuffer<B1>>>) -> Self {
		let secondaries = secondaries.into_iter();
		let (lower, upper) = secondaries.size_hint();
		let mut secondary_vks = Vec::with_capacity(upper.unwrap_or(lower));
		for sec in secondaries {
			secondary_vks.push(sec.vk);
			self.resources.push(Resource::CommandBuffer(sec));
		}

		unsafe { self.pool.device.vk.cmd_execute_commands(self.vk, &secondary_vks) };
		self
	}
}

pub struct InheritanceInfo {
	pub render_pass: Arc<RenderPass>,
	pub subpass: u32,
	pub framebuffer: Option<Arc<Framebuffer>>,
}

pub struct CommandBuffer<SEC: Bit> {
	pub(crate) pool: Arc<CommandPool>,
	vkpool: vk::CommandPool,
	pub(crate) vk: vk::CommandBuffer,
	_resources: Vec<Resource>,
	sec: PhantomData<SEC>,
}
impl<SEC: Bit> CommandBuffer<SEC> {
	unsafe fn from_vk(
		pool: Arc<CommandPool>,
		vkpool: vk::CommandPool,
		vk: vk::CommandBuffer,
		resources: Vec<Resource>,
	) -> Arc<Self> {
		Arc::new(Self { pool, vkpool, vk, _resources: resources, sec: PhantomData })
	}
}
impl<SEC: Bit> Drop for CommandBuffer<SEC> {
	fn drop(&mut self) {
		let mut free_lock = self.pool.free.lock().unwrap();
		let free = free_lock.get_mut(&self.vkpool).unwrap();
		let cmds = if SEC::BOOL { &mut free.secondary } else { &mut free.primary };
		cmds.push(self.vk);
	}
}
