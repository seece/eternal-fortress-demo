#version 460
// the local thread block size; the program will be ran in sets of 16 by 16 threads.
layout(local_size_x = 16, local_size_y = 16) in;

// simple pseudo-RNG based on the jenkins hash mix function
uvec4 rndseed;
void jenkins_mix()
{
	rndseed.x -= rndseed.y; rndseed.x -= rndseed.z; rndseed.x ^= rndseed.z >> 13;
	rndseed.y -= rndseed.z; rndseed.y -= rndseed.x; rndseed.y ^= rndseed.x << 8;
	rndseed.z -= rndseed.x; rndseed.z -= rndseed.y; rndseed.z ^= rndseed.y >> 13;
	rndseed.x -= rndseed.y; rndseed.x -= rndseed.z; rndseed.x ^= rndseed.z >> 12;
	rndseed.y -= rndseed.z; rndseed.y -= rndseed.x; rndseed.y ^= rndseed.x << 16;
	rndseed.z -= rndseed.x; rndseed.z -= rndseed.y; rndseed.z ^= rndseed.y >> 5;
	rndseed.x -= rndseed.y; rndseed.x -= rndseed.z; rndseed.x ^= rndseed.z >> 3;
	rndseed.y -= rndseed.z; rndseed.y -= rndseed.x; rndseed.y ^= rndseed.x << 10;
	rndseed.z -= rndseed.x; rndseed.z -= rndseed.y; rndseed.z ^= rndseed.y >> 15;
}
void srand(uint A, uint B, uint C) { rndseed = uvec4(A, B, C, 0); jenkins_mix(); jenkins_mix(); }
float rand()
{
	if (0 == rndseed.w++ % 3) jenkins_mix();
	return float((rndseed.xyz = rndseed.yzx).x) / pow(2., 32.);
}

const int MATERIAL_SKY = 0;
const int MATERIAL_OTHER = 1;

// uniform variables are global from the glsl perspective; you set them in the CPU side and every thread gets the same value
uniform int source;
uniform int frame;
uniform float secs;

layout(r32f) uniform image2D zbuffer;
layout(rgba16f) uniform image2DArray samplebuffer;

struct CameraParams {
    vec3 pos;
	float padding;
	vec3 dir;
	float zoom;
	vec3 up;
	float padding2;
	vec3 right;
	float padding3;
};

layout(std140) uniform cameraArray {
	CameraParams cameras[2];
};

vec2 getThreadUV(uvec3 id) {
	return vec2(id.xy) / 1024.0;
}

// mandelbox distance function by Rrola (Buddhi's distance estimation)
// http://www.fractalforums.com/index.php?topic=2785.msg21412#msg21412

const float SCALE = 2.7;
const float MR2 = 0.01;
const int mbox_iters = 7;
vec4 mbox_scalevec = vec4(SCALE, SCALE, SCALE, abs(SCALE)) / MR2;
float mbox_C1 = abs(SCALE-1.0), mbox_C2 = pow(abs(SCALE), float(1-mbox_iters));

float mandelbox(vec3 position) {

	// distance estimate
	vec4 p = vec4(position.xyz, 1.0), p0 = vec4(position.xyz, 1.0);  // p.w is knighty's DEfactor
	for (int i=0; i<mbox_iters; i++) {
		p.xyz = clamp(p.xyz, -1.0, 1.0) * 2.0 - p.xyz;  // box fold: min3, max3, mad3
		float r2 = dot(p.xyz, p.xyz);  // dp3
		p.xyzw *= clamp(max(MR2/r2, MR2), 0.0, 1.0);  // sphere fold: div1, max1.sat, mul4
		p.xyzw = p*mbox_scalevec + p0;  // mad4
	}
	return (length(p.xyz) - mbox_C1) / p.w - mbox_C2;
}

float scene(vec3 p, out int material) {
	material = MATERIAL_OTHER;
	return mandelbox(p);
}

vec3 evalnormal(vec3 p) {
	vec2 e=vec2(1e-5, 0.f);
	int m;
	return normalize(vec3(
				scene(p + e.xyy,m) - scene(p - e.xyy,m),
				scene(p + e.yxy,m) - scene(p - e.yxy,m),
				scene(p + e.yyx,m) - scene(p - e.yyx,m)
				));
}

void getCameraProjection(CameraParams cam, vec2 uv, out vec3 outPos, out vec3 outDir) {
	uv /= cam.zoom;
	// vec3 right = cross(cam.dir, vec3(0., 1., 0.));
	// vec3 up = cross(cam.dir, right);
	outPos = cam.pos + cam.dir + (uv.x - 0.5) * cam.right + (uv.y - 0.5) * cam.up;
	outDir = normalize(outPos - cam.pos);
}

// absolute error (measured with euclidian length) seems to be less than 0.003
float packJitter(vec2 jitter)
{
    vec2 a = jitter + vec2(0.5);
    a *= 255;
    uvec2 b = uvec2(a);
    uint c = (b.x << 8) | b.y;
    return uintBitsToFloat(c);
}

vec2 unpackJitter(float d)
{
    uint c = floatBitsToUint(d);
    uvec2 b = uvec2((c >> 8) & 0xff, c & 0xff);
    vec2 a = vec2(b) / 255;
    return a - vec2(0.5);
}

void main() {
    uint seed = uint(gl_GlobalInvocationID.x) + uint(gl_GlobalInvocationID.y);
    seed = uint(rand()*9999 + frame);

    int samplesPerPixel = imageSize(samplebuffer).z;
    float minDepth = 1e20;
    vec3 accum = vec3(0.);

    for (int sample_id=0; sample_id < samplesPerPixel; sample_id++)
    {
        vec2 jitter = vec2(rand(), rand()) - vec2(0.5, 0.5);

        vec2 uv = getThreadUV(gl_GlobalInvocationID);
        jitter /= imageSize(zbuffer).xy;
        uv += jitter;

        CameraParams cam = cameras[1];
        vec3 p, dir;
        getCameraProjection(cam, uv, p, dir);

        int hitmat = MATERIAL_SKY;
        int i;

        for (i=0;i<300;i++) {
            int mat;
            float d = scene(p, mat);
            if (d < 1e-5) {
                hitmat = mat;
                break;
            }
            p += d * dir;
        }

        vec3 color;

        float depth = 1e10;
        switch (hitmat) {
            case MATERIAL_SKY:
                color = vec3(0., 0.2*abs(dir.y), 0.5 - dir.y);
                break;
            case MATERIAL_OTHER:
                depth = length(p - cam.pos);

                vec3 normal = evalnormal(p);
                vec3 to_camera = normalize(cam.pos - p);
                vec3 to_light = normalize(vec3(-0.5, -1.0, 0.7));
                float shine = 0.5+0.5*dot(normal, to_light);
                shine = pow(shine, 16.);
                color = vec3(shine);
                break;
        }

        accum += color;
        minDepth = min(minDepth, depth);

        imageStore(samplebuffer, ivec3(gl_GlobalInvocationID.xy, sample_id), vec4(color, packJitter(jitter)));
    }

    imageStore(zbuffer, ivec2(gl_GlobalInvocationID.xy), vec4(minDepth));
}

