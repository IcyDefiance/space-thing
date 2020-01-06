#version 450
#define USE_VULKAN
//#version 130

//#define IterationDebugView

#ifdef USE_VULKAN
layout(location = 0) out vec4 out_color;
layout(binding = 0) uniform sampler3D voxels;
layout(binding = 1) uniform sampler3D mats;
layout(binding = 2) uniform sampler2D blocks;

layout(push_constant) uniform PushConsts {
	vec3 cam_pos;
	vec4 cam_rot;
} pc;
const vec2 iResolution = vec2(1440.0, 810.0);
#define CAMROT pc.cam_rot
#define CAMPOS pc.cam_pos
#define PIXOUT out_color
#else
uniform sampler3D voxels;
uniform sampler3D mats;
uniform vec3 cam_pos;
uniform vec4 cam_rot;
uniform vec2 iResolution;
#define CAMROT cam_rot
#define CAMPOS cam_pos
#define PIXOUT gl_FragColor
#endif

#define RayIntersectQuality 1024
#define RayRefineQuality 256
#define ShadowQuality 256
#define AOQuality 32

const float ViewDistance = 256.0;
const vec3 AmbientLight = vec3(0.25, 0.5, 0.75);
const float MinStepSize = 0.01;
const float GridSize = 0.25;
const vec2 AOSize = vec2(2.0, GridSize * GridSize);
const float tiny = 0.001;

vec3 quat_mul(vec4 quat, vec3 vec) {
	return cross(quat.xyz, cross(quat.xyz, vec) + vec * quat.w) * 2.0 + vec;
}

float sdBox(vec3 p, vec3 b) {
	vec3 q = abs(p) - b;
	return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float F(vec3 pos) {
	const float range = 10.0;
	const vec3 BlockSize = vec3(16.0, 16.0, 256.0);
	vec3 tc = ((pos+0.5) / BlockSize) + vec3(0.5);
	vec3 ej = vec3(1.0/BlockSize);
	tc = clamp(tc, ej, 1.0 - ej);
	return texture(voxels, tc).r * (range + 1.0) - 1.0;
}

#ifdef IterationDebugView
vec2 shadowRay(vec3 pos, vec3 dir) {
	float iters = 0.0;
#else
float shadowRay(vec3 pos, vec3 dir) {
#endif
	const float sharpness = 13.0;
	float s = 1.0;
	float t = GridSize * GridSize;
	float ph = tiny;
	for(int i=0;i<ShadowQuality;i++) {
		float h = F(pos + dir * t);
#ifdef IterationDebugView
		iters += 1.0;
        if (h < tiny) return vec2(0.0, iters);
#else
        if (h < tiny) return 0.0;
#endif
        float y = h*h / (2.0*ph);
        float d = sqrt(h*h - y*y);
        s = min(s, sharpness*d/max(0.0, t-y));
        ph = h;
        t += h;
        if (t > ViewDistance) break;
	}
#ifdef IterationDebugView
	return vec2(max(s, 0.0), iters);
#else
	return max(s, 0.0);
#endif
}

vec3 tonemap(vec3 color) {
	color /= 1.0 + color;
	return sqrt(color);
}

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
#ifdef IterationDebugView
vec2 ao(vec3 p, vec3 n) {
	float iters = 0.0;
#else
float ao(vec3 p, vec3 n) {
#endif
	const float Qi = 1.0 / float(AOQuality);
	vec2 ao = vec2(0.0);
	for(int i=0;i<AOQuality;i++){
		vec2 l = randf(i) * AOSize;
		vec3 dx = normalize(n + randomHemisphereDir(n, l.x) * (1.0 - Qi)) * l.x;
		vec3 dy = normalize(n + randomHemisphereDir(n, l.y) * (1.0 - Qi)) * l.y;
		ao += l - max(vec2(F(p + dx), F(p + dy)), vec2(0.0));
#ifdef IterationDebugView
		iters += 2.0;
#endif
	}
	ao = clamp(1.0 - 2.0 * ao * Qi / AOSize, 0.0, 1.0);
#ifdef IterationDebugView
	return vec2(ao.x * sqrt(ao.y), iters);
#else
	return ao.x * sqrt(ao.y);
#endif
}

vec3 T(vec2 uv) {
	uv *= 3.14159 * 1.618;
	return vec3(sin(uv.x)*sin(uv.y)*0.5 + 0.5);
}

void main() {
#ifdef IterationDebugView
	float iters = 0.0;
#endif
#ifdef USE_VULKAN
	vec3 dir = quat_mul(CAMROT, normalize(vec3(
		(2.0*gl_FragCoord.x - iResolution.x) / iResolution.y,
		1.0,
		1.0 - 2.0*gl_FragCoord.y / iResolution.y
	)));
#else
	vec3 dir = quat_mul(CAMROT, normalize(vec3(
		(2.0*gl_FragCoord.x - iResolution.x) / iResolution.y,
		1.0,
		1.0 - 2.0*(iResolution.y - gl_FragCoord.y) / iResolution.y
	)));
#endif
	vec3 pos = CAMPOS;
	float t = tiny;
	for(int i=0;i<RayIntersectQuality;i++) {
		float d = F(pos + dir * t);
#ifdef IterationDebugView
		iters += 1.0;
#endif
		if (d < 0.0) break;
		t += max(d, MinStepSize);
		if (t > ViewDistance) break;
	}
	float dInside = F(pos + dir * t);
	bool miss = (isnan(dInside) || abs(dInside) > GridSize);
#ifdef IterationDebugView
	iters += 1.0;
#else
	if (miss) {
		//discard;
		PIXOUT = vec4(AmbientLight, 0.0);
		return;
	}
#endif

	float dOutside = F(pos + dir * (t - GridSize));
#ifdef IterationDebugView
	if (!miss) iters += 1.0;
#endif
	t += GridSize * dInside / (dOutside - dInside);
	pos += dir * t;
	for(int i=0;i<RayRefineQuality;i++) {
		float d = F(pos);
#ifdef IterationDebugView
		if (!miss) iters += 1.0;
#endif
		pos += dir * d;
		if (abs(d) < tiny) break;
	}

	const float smoothness = 0.0;
	float k = mix(tiny * 8.0, GridSize * 0.5, smoothness);
	//vec3 nor = normalize(k.xyy*F(pos + k.xyy) + k.yyx*F(pos + k.yyx) + k.yxy*F(pos + k.yxy) + k.xxx*F(pos + k.xxx));
	vec3 nor = normalize(vec3(
		F(pos + vec3(k, 0.0, 0.0)) - F(pos - vec3(k, 0.0, 0.0)),
		F(pos + vec3(0.0, k, 0.0)) - F(pos - vec3(0.0, k, 0.0)),
		F(pos + vec3(0.0, 0.0, k)) - F(pos - vec3(0.0, 0.0, k))
	));

	vec3 triplane = nor * nor;
	triplane *= triplane;
	// triplane *= triplane; // uncomment this for a tighter fade between sides
	vec3 colorX = T(pos.yz);
	vec3 colorY = T(pos.xz);
	vec3 colorZ = T(pos.xy);
	vec3 color = (colorX * triplane.x + colorY * triplane.y + colorZ * triplane.z) / (triplane.x + triplane.y + triplane.z);

#ifdef IterationDebugView
	vec2 aoResult = ao(pos, nor);
	vec3 light = AmbientLight * aoResult.x;
	if (!miss) iters += aoResult.y;
#else
	vec3 light = AmbientLight * ao(pos, nor);
#endif

	// *** add lights here ***
	vec3 lightDir1 = normalize(vec3(3.0, -4.0, 5.0));
#ifdef IterationDebugView
	vec2 shadowResult = shadowRay(pos, lightDir1);
	light += vec3(1.0, 1.0, 1.0) * max(0.0, dot(nor, lightDir1)) * shadowResult.x;
	if (!miss) iters += shadowResult.y;
#else
	light += vec3(1.0, 1.0, 1.0) * max(0.0, dot(nor, lightDir1)) * shadowRay(pos, lightDir1);
#endif
	//vec3 lightDir2 = normalize(vec3(-5.0, 15.0, 2.0));
	//light += vec3(1.0, 0.9, 0.8) * max(0.0, dot(nor, lightDir2)) * shadowRay(pos, lightDir2);

#ifdef IterationDebugView
	const float band1 = 32.0;
    const float band2 = 256.0;
    const float band3 = 1024.0;
    if (iters > band2) {
        float ss = smoothstep(band2, band3, iters);
        PIXOUT = vec4(mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), ss), 0.0);
    } else if (iters > band1) {
        float ss = smoothstep(band1, band2, iters);
        PIXOUT = vec4(mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0), ss), 0.0);
    } else {
		float ss = smoothstep(0.0, band1, iters);
        PIXOUT = vec4(mix(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), ss), 0.0);
    }
#else
	PIXOUT = vec4(tonemap(color * light), 0.0);
#endif
}
