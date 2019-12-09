use shaderc::{Compiler, ShaderKind};
use std::{
	fs::{create_dir, File},
	io::prelude::*,
	path::Path,
};

fn main() {
	create_dir("build").ok();
	build_shader("src/gfx/shaders/shader.vert", "build/shader.vert.spv", ShaderKind::Vertex);
	build_shader("src/gfx/shaders/shader.frag", "build/shader.frag.spv", ShaderKind::Fragment);
}

fn build_shader(input: &str, output: &str, kind: ShaderKind) {
	let input = Path::new(input);
	let output = Path::new(output);

	let mut file = File::open(input).unwrap();
	let mut source = String::new();
	file.read_to_string(&mut source).unwrap();

	let mut compiler = Compiler::new().unwrap();
	let binary_result =
		compiler.compile_into_spirv(&source, kind, input.file_name().unwrap().to_str().unwrap(), "main", None).unwrap();

	let mut file = File::create(output).unwrap();
	file.write_all(binary_result.as_binary_u8()).unwrap();
}
