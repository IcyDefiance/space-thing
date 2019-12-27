use ash::vk;
use nalgebra::Vector4;
use std::{collections::HashMap, sync::Arc};

type Color = Vector4<u8>;

pub trait Node {}

pub struct Document {
	body: Vec<Arc<dyn Node>>,
	rect: vk::Rect2D,
}

pub struct DivElement {
	children: Vec<Arc<dyn Node>>,
	style: Styles,
}
impl Node for DivElement {}

struct Styles {
	map: HashMap<StyleName, StyleValue>,
}
impl Styles {
	fn background_color(&self) -> Option<Color> {
		self.map.get(&StyleName::BackgroundColor).map(|x| unsafe { x.color })
	}

	fn set_background_color(&mut self, color: Color) {
		self.map.insert(StyleName::BackgroundColor, StyleValue { color });
	}
}

#[derive(PartialEq, Eq, Hash)]
enum StyleName {
	BackgroundColor,
}

union StyleValue {
	color: Color,
}
