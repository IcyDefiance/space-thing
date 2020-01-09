#version 450
#define USE_VULKAN

#ifdef USE_VULKAN
layout(location = 0) out vec4 out_color;
layout(binding = 0) uniform sampler3D voxels;
layout(binding = 1) uniform sampler3D mats;
layout(binding = 2) uniform sampler2D blocks[2];
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
uniform sampler2D blocks[2];
uniform vec3 cam_pos;
uniform vec4 cam_rot;
uniform vec2 iResolution;
#define CAMROT cam_rot
#define CAMPOS cam_pos
#define PIXOUT gl_FragColor
#endif

#define RayIntersectQuality 1024
#define RayRefineQuality 64
#define ShadowQuality 32
#define AOQuality 32

const float TextureResolution = 512.0;
const float ViewDistance = 1600.0;
const vec3 AmbientLight = vec3(0.25, 0.5, 0.75);
const float MinStepSize = 0.01;
const float GridSize = 0.25;
const vec2 AOSize = vec2(2.0, GridSize * GridSize);
const float NormalSmoothing = 0.0;
const float tiny = 0.001;

vec3 quat_mul(vec4 quat, vec3 vec) {
	return cross(quat.xyz, cross(quat.xyz, vec) + vec * quat.w) * 2.0 + vec;
}

float sdBox(vec3 p, vec3 b) {
	vec3 q = abs(p) - b;
	return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float F(vec3 pos) {
	float d = dot(pos, vec3(0.0, 0.0, 1.0)) - 128.0;
	d = min(d, sdBox(pos - vec3(5.5,2.5,128.5), vec3(0.5)));
	d = min(d, sdBox(pos - vec3(3.5,2.5,128.5), vec3(0.5)));
	d = min(d, sdBox(pos - vec3(2.5,2.5,128.5), vec3(0.5)));
	d = min(d, sdBox(pos - vec3(3.5,3.5,128.5), vec3(0.5)));
	//if (gl_FragCoord.x < iResolution.x/2.0) return d;

	const float range = 10.0;
	const vec3 BlockSize = vec3(16.0, 16.0, 256.0);
	vec3 tc = (pos + vec3(GridSize * 0.5)) / BlockSize;
	tc = clamp(tc, 1.0 / BlockSize, 1.0 - 1.0 / BlockSize);
	return min(pos.z, texture(voxels, tc).r * (range + 1.0) - 1.0);
}
float getTextureID(vec3 pos, vec3 nor) {
	for(int i=0;i<4;i++) pos -= nor * (F(pos) + MinStepSize * 2.0);
	const vec3 BlockSize = vec3(16.0, 16.0, 256.0);
	vec3 tc = (pos - nor * GridSize * MinStepSize) / BlockSize;
	return texture(mats, tc).r * 255.0;
}
vec4 T(vec2 uv, vec2 dx, vec2 dy, float i) {
	uv.x = mod(uv.x, 1.0) + i;
	uv.x *= 0.125;
	dx.x *= 0.125;
	dy.x *= 0.125;

	float margin = 0.5 / TextureResolution;
	margin = min(margin, length(dx));
	margin = max(margin, 0.05 / TextureResolution);
	uv.x = clamp(uv.x, i*0.125 + margin, (i+1.0)*0.125 - margin);

	return textureGrad(blocks[0], uv, dx, dy);
}
vec4 T2(vec2 uv, vec2 dx, vec2 dy, float i) {
	uv.x = mod(uv.x, 1.0) + i;
	uv.x *= 0.125;
	dx.x *= 0.125;
	dy.x *= 0.125;
	float margin = 0.5 / TextureResolution;
	margin = min(margin, length(dx));
	margin = max(margin, 0.05 / TextureResolution);
	uv.x = clamp(uv.x, i*0.125 + margin, (i+1.0)*0.125 - margin);

	return textureGrad(blocks[1], uv, dx, dy);
}

float shadowRay(vec3 pos, vec3 dir) {
	const float sharpness = 13.0;
	float s = 1.0;
	float t = GridSize * GridSize;
	float ph = tiny;
	for(int i=0;i<ShadowQuality;i++) {
		float h = F(pos + dir * t);
        if (h < tiny) return 0.0;
        float y = h*h / (2.0*ph);
        float d = sqrt(h*h - y*y);
        s = min(s, sharpness*d/max(0.0, t-y));
        ph = h;
        t += h;
        if (t > ViewDistance) break;
	}
	return max(s, 0.0);
}

vec3 tonemap(vec3 color) {
	color /= 1.0 + color;
	return sqrt(color);
}

float hash(float p) {
	p = fract(p * sqrt(5.0)) + sqrt(6.0);
	p += sin(p * 1337.0);
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
		vec2 l = vec2(randf(i), randf(i+123)) * AOSize;
		vec3 dx = normalize(n + randomHemisphereDir(n, l.x) * (1.0 - Qi)) * l.x;
		vec3 dy = normalize(n + randomHemisphereDir(n, l.y) * (1.0 - Qi)) * l.y;
		ao += l - max(vec2(F(p + dx), F(p + dy)), vec2(0.0));
	}
	ao = clamp(1.0 - 2.0 * ao * Qi / AOSize, 0.0, 1.0);
	return ao.x * ao.y;
}

vec4 raytrace(vec3 pos, vec3 dir) {
	float t = tiny;
	for(int i=0;i<RayIntersectQuality;i++) {
		float d = F(pos + dir * t);
		if (d < 0.0) break;
		t += max(d, MinStepSize);
		if (t > ViewDistance) break;
	}
	float dInside = F(pos + dir * t);
	bool miss = (isnan(dInside) || abs(dInside) > GridSize);
	if (miss) return vec4(0.0, 0.0, 0.0, -1.0);

	float dOutside = F(pos + dir * (t - GridSize));
	t += GridSize * dInside / (dOutside - dInside);
	pos += dir * t;
	for(int i=0;i<RayRefineQuality;i++) {
		float d = F(pos);
		pos += dir * d;
		t += d;
		//if (abs(d) < tiny) break;
	}
	return vec4(pos, t);
}
vec3 getNormal(vec3 pos) {
	float k = mix(tiny * 8.0, GridSize * 0.5, NormalSmoothing);
	//vec3 nor = normalize(k.xyy*F(pos + k.xyy) + k.yyx*F(pos + k.yyx) + k.yxy*F(pos + k.yxy) + k.xxx*F(pos + k.xxx));
	vec3 nor = normalize(vec3(
		F(pos + vec3(k, 0.0, 0.0)) - F(pos - vec3(k, 0.0, 0.0)),
		F(pos + vec3(0.0, k, 0.0)) - F(pos - vec3(0.0, k, 0.0)),
		F(pos + vec3(0.0, 0.0, k)) - F(pos - vec3(0.0, 0.0, k))
	));
	return nor;
}

vec4 getTexture(float i, vec3 pos, vec3 nor, vec3 dpdx, vec3 dpdy) {
	vec3 triplane = nor * nor;
	triplane *= triplane;
	triplane *= triplane;
	triplane *= triplane;
	pos -= vec3(GridSize);
	vec4 colorX = T(pos.yz, dpdx.yz, dpdy.yz, i);
	vec4 colorY = T(pos.xz * vec2(1.0, -1.0), dpdx.xz * vec2(1.0, -1.0), dpdy.xz * vec2(1.0, -1.0), i);
	vec4 colorZ = T(pos.xy, dpdx.xy, dpdy.xy, i);
	vec4 color = (colorX * triplane.x + colorY * triplane.y + colorZ * triplane.z) / (triplane.x + triplane.y + triplane.z);
	return color;
}

void main() {
	float dx0 = (2.0*gl_FragCoord.x - iResolution.x) / iResolution.y;
	float dx1 = (2.0*(gl_FragCoord.x+1.0) - iResolution.x) / iResolution.y;
#ifdef USE_VULKAN
	float dy0 = 1.0 - 2.0*gl_FragCoord.y / iResolution.y;
	float dy1 = 1.0 - 2.0*(gl_FragCoord.y+1.0) / iResolution.y;
#else
	float dy0 = 1.0 - 2.0*(iResolution.y - gl_FragCoord.y) / iResolution.y;
	float dy1 = 1.0 - 2.0*(iResolution.y - (gl_FragCoord.y+1.0)) / iResolution.y;
#endif
	vec3 dir = quat_mul(CAMROT, normalize(vec3(dx0, 1.0, dy0)));
	vec3 rdx = quat_mul(CAMROT, normalize(vec3(dx1, 1.0, dy0)));
	vec3 rdy = quat_mul(CAMROT, normalize(vec3(dx0, 1.0, dy1)));

	// primary ray
	vec4 ray = raytrace(CAMPOS, dir);
	if (ray.w < 0.0) {
		PIXOUT = vec4(AmbientLight, 0.0);
		return;
	}
	vec3 pos = ray.xyz;
	vec3 nor = getNormal(pos);
	vec3 dpdx = ray.w * (rdx*dot(dir, nor) / dot(rdx, nor) - dir);
    vec3 dpdy = ray.w * (rdy*dot(dir, nor) / dot(rdy, nor) - dir);
	float tex = getTextureID(pos, nor);
	vec3 color = getTexture(tex, pos, nor, dpdx, dpdy).rgb;

	// lights
	vec3 light = AmbientLight * ao(pos, nor);
	vec3 lightDir1 = normalize(vec3(3.0, -4.0, 5.0));
	vec3 lightColor1 = vec3(1.0, 1.0, 1.0);
	light += lightColor1 * max(0.0, dot(nor, lightDir1)) * shadowRay(pos, lightDir1);

	color *= light;

	/* mirror reflection
	float dfac = 1.0 - dot(dir, -nor);
	float dfac2 = dfac * dfac;
	float rfac = mix(0.01, 1.0, dfac * dfac2 * dfac2) * 0.5;
	vec3 rvec = reflect(dir, nor);
	vec4 rray = raytrace(pos + rvec * tiny, rvec);
	if (rray.w < 0.0) {
		color = mix(color, AmbientLight, rfac);
	} else {
		vec3 rpos = rray.xyz;
		vec3 rnor = getNormal(rpos);
		vec3 rdpdx = dFdx(rpos);
		vec3 rdpdy = dFdy(rpos);
		float rtex = getTextureID(rpos, rnor);
		vec3 rcol = getTexture(rtex, rpos, rnor, rdpdx, rdpdy).rgb;
		vec3 rlight = AmbientLight * ao(rpos, rnor);
		rlight += lightColor1 * max(0.0, dot(rnor, lightDir1)) * shadowRay(rpos, lightDir1);
		color = mix(color, rlight * rcol, rfac);
	}
	//*/

	PIXOUT = vec4(tonemap(color), 0.0);
}
