#version 450

#define AOQuality 8

layout(location = 0) out vec4 out_color;

layout(binding = 0) uniform sampler smpl;
layout(binding = 1) uniform texture3D voxels[2];

const vec2 AOSize = vec2(3.14159, 2.0 * 3.14159 * 3.14159);
const float ViewDistance = 1000.0;
const vec3 AmbientLight = vec3(0.25, 0.5, 0.75);
const float MinStepSize = 0.01;
const float GridSize = 1.0;
const float tiny = 0.001;

const vec4 cam_proj = vec4(0.5625, 1.0, -1.002002, -1.001001);
const vec3 cam_pos = vec3(0.0, 1.618, 4.0);
const vec4 cam_rot = vec4(0.0, 0.0, 0.0, 1.0);

vec3 quat_mul(vec4 quat, vec3 vec) {
	return cross(quat.xyz, cross(quat.xyz, vec) + vec * quat.w) * 2.0 + vec;
}

float sdBox(vec3 p, vec3 b) {
	vec3 q = abs(p) - b;
	return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float F(vec3 pos) {
	return texture(sampler3D(voxels[0], smpl), pos).r;
	// float d = dot(pos, vec3(0.0, 1.0, 0.0));
	// for(int i=0;i<5;i++) {
	// 	vec3 boxPos = vec3(0.0, float(2*i) + 0.5, 0.0);
	// 	float v = sdBox(pos - boxPos, vec3(0.5));
	// 	d = min(d, v);
	// }
	// return d;
}

float shadowRay(vec3 pos, vec3 dir) {
	const float sharpness = 4.0;
	float s = 1.0;
	float t = 0.5 * GridSize;
	for(int i=0;i<64;i++) {
		vec3 p = pos + dir * t;
		float d = F(p);
		s = min(s, sharpness * d / t);
		t += max(d, MinStepSize);
	}
	return max(s, 0.0);
}

vec3 tonemap(vec3 color) {
	color /= 1.0 + color;
	return sqrt(color);
}

#if AOQuality == 0
float ao(vec3 p, vec3 n) {
	float dist = 0.618;
	float occ = 1.0;
	for (int i=0;i<4;i++) {
		occ = min(occ, F(p + dist * n) / dist);
		dist *= 2.0;
	}
	return max(occ, 0.0);
}
#else
float hash(float p) {
	p = fract(p * sqrt(5.0)) + sqrt(6.0);
	return fract(p * sqrt(7.0));
}
vec2 hash2(float p) {
	return vec2(hash(p + sqrt(2.0)), hash(p + sqrt(3.0)));
}
float randf(int i) {
	return hash(float(i));
}
vec3 randomSphereDir(vec2 rnd) {
	float s = rnd.x*3.14159*2.0;
	float t = rnd.y*2.0 - 1.0;
	return vec3(sin(s), cos(s), t) / sqrt(1.0 + t * t);
}
vec3 randomHemisphereDir(vec3 dir, float i) {
	vec3 v = randomSphereDir(hash2(i));
	return v * sign(dot(v, dir));
}
float ao(vec3 p, vec3 n) {
	const float Qi = 1.0 / float(AOQuality);
	vec2 ao = vec2(0.0);
	for(int i=0;i<AOQuality;i++){
		vec2 l = randf(i) * AOSize;
		vec3 dx = normalize(n + randomHemisphereDir(n, l.x) * (1.0 - Qi)) * l.x;
		vec3 dy = normalize(n + randomHemisphereDir(n, l.y) * (1.0 - Qi)) * l.y;
		ao += l - max(vec2(F(p + dx), F(p + dy)), vec2(0.0));
	}
	ao = clamp(1.0 - 2.0 * ao * Qi / AOSize, 0.0, 1.0);
	float r = 0.618 * (ao.x + ao.y);
	return min(r * r, 1.0);
}
#endif

void main() {
	const vec2 iResolution = vec2(1440.0, 810.0); // FIXME
	vec3 dir = quat_mul(cam_rot, normalize(vec3((2.0*gl_FragCoord.xy-iResolution.xy)/iResolution.y, -1.0)));
	vec3 pos = cam_pos;
	float t = tiny;
	for(int i=0;i<128;i++) {
		// use low-res volume texture here
		float d = F(pos + dir * t);
		if (d < 0.0) break;
		t += max(d, MinStepSize);
		if (t > ViewDistance) break;
	}
	float dInside = F(pos + dir * t);
	if (isnan(dInside) || abs(dInside) > GridSize) discard;
	float dOutside = F(pos + dir * (t - GridSize));
	t += GridSize * dInside / (dOutside - dInside);
	pos += dir * t;
	for(int i=0;i<128;i++) {
		// use high-res volume texture here
		float d = F(pos);
		pos += dir * d;
		if (abs(d) < tiny) break;
	}

	const vec2 k = vec2(1.0, -1.0);
	vec3 nor = normalize(k.xyy*F(pos + k.xyy*tiny) + k.yyx*F(pos + k.yyx*tiny) + k.yxy*F(pos + k.yxy*tiny) + k.xxx*F(pos + k.xxx*tiny));

	// add texturing here
	vec3 color = vec3(0.5);

	vec3 light = AmbientLight * ao(pos, nor);

	// *** add lights here ***
	vec3 lightDir1 = normalize(vec3(5.0, 4.0, 3.0));
	light += vec3(1.0, 1.0, 1.0) * max(0.0, dot(nor, lightDir1)) * shadowRay(pos, lightDir1);
	//vec3 lightDir2 = normalize(vec3(-5.0, 15.0, 2.0));
	//light += vec3(1.0, 0.9, 0.8) * max(0.0, dot(nor, lightDir2)) * shadowRay(pos, lightDir2);

	out_color = vec4(tonemap(color * light), 0.0);
}
