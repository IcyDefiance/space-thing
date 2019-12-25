#version 450

layout(location = 0) in vec2 in_pos;

layout(location = 0) out vec4 out_color;

vec4 cam_proj = vec4(0.5625, 1, -1.002002, -1.001001);
vec3 cam_pos = vec3(0, -5, 0);
vec4 cam_rot = vec4(0, 0, 0, 1);
float sphere_radius = 1;

float F(vec3 pos) {
	return length(pos) - 1;
}

vec3 perspective(vec4 proj, vec3 pos) {
	return vec3(pos.xy * proj.xy, pos.z * proj.z + proj.w);
}

vec3 quat_mul(vec4 quat, vec3 vec) {
	return cross(quat.xyz, cross(quat.xyz, vec) + vec * quat.w) * 2.0 + vec;
}

void main() {
	vec3 cam_dir_cs = quat_mul(cam_rot, vec3(0, 1, 0));
	vec3 cam_dir_es = normalize(cam_dir_cs + vec3(in_pos.x, 0, in_pos.y));
	vec2 in_pos_nor = (in_pos + 1) / 2;
	vec2 px = vec2(1) * in_pos_nor / gl_FragCoord.xy;

	float distance;
	vec3 pos = cam_pos;
	for (int i = 0; i < 32; ++i) {
		distance = F(pos);
		pos += cam_dir_es * distance;
	}
	float depth = length(pos - cam_pos);
	if (distance > length(px * depth)) {
		discard;
	}

	out_color = vec4(0.8, 0.8, 0.8, 1.0);
	// output normalized depth
}
