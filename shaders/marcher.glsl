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

const int MATERIAL_SKY = -1;
const int MATERIAL_OTHER = 1;

// uniform variables are global from the glsl perspective; you set them in the CPU side and every thread gets the same value
uniform int source;
uniform int frame;
uniform float secs;

layout(r32f) uniform image2D zbuffer;
layout(r8) uniform image2D edgebuffer;
layout(rgba16f) uniform image2DArray samplebuffer;
layout(rg8) uniform image2DArray jitterbuffer;
uniform ivec3 noiseOffset;
uniform sampler2DArray noiseTextures;

vec3 getNoise(ivec2 coord)
{
    return texelFetch(noiseTextures,
            ivec3((coord.x + noiseOffset.x) % 64, (coord.y + noiseOffset.y) % 64, noiseOffset.z),0).rgb;
}

struct CameraParams {
    vec3 pos;
    float padding0;
    vec3 dir;
    float nearplane;
    vec3 up;
    float padding2;
    vec3 right;
    float padding3;
};

struct RgbPoint {
    vec4 xyzw;
    vec4 rgba;
};

uniform int pointBufferMaxElements;
uniform int jumpBufferMaxElements;

layout(std140) uniform cameraArray {
    CameraParams cameras[2];
};

layout(std430) buffer pointBufferHeader {
    int currentWriteOffset;
    int pointsSplatted;
    int nextRayIndex;
};

layout(std430) buffer pointBuffer {
    RgbPoint points[];
};

layout(std430) buffer jumpBuffer {
    float jumps[];
};

layout(std430) buffer debugBuffer {
    int debug_i;
    int debug_parent;
    int debug_size;
    int debug_b;
    int debug_start;
    int debug_parent_size;
    float debug_pixelRadius;
    float debug_zdepth;
    float debug_parentDepth;
};

float PIXEL_RADIUS;
float PROJECTION_PLANE_DIST;
float NEAR_PLANE;

// This factor how many pixel radiuses of screen space error do we allow
// in the "near geometry snapping" at the end of "march" loop. Without it
// the skybox color leaks through with low grazing angles.
const float SNAP_INFLATE_FACTOR = 3.;
const float DISTANCE_INFLATE_FACTOR = 1.01;

// Ray's maximum travel distance.
const float MAX_DISTANCE = 1e9;

// mandelbox distance function by Rrola (Buddhi's distance estimation)
// http://www.fractalforums.com/index.php?topic=2785.msg21412#msg21412

const float SCALE = 2.7;
const float MR2 = 0.01;
vec4 mbox_scalevec = vec4(SCALE, SCALE, SCALE, abs(SCALE)) / MR2;
float mbox_C1 = abs(SCALE-1.0);

int mandelbox_material(vec3 position, int iters=7) {
    float mbox_C2 = pow(abs(SCALE), float(1-iters));
	// distance estimate
	vec4 p = vec4(position.xyz, 1.0), p0 = vec4(position.xyz, 1.0);  // p.w is knighty's DEfactor
	for (int i=0; i<iters; i++) {
		p.xyz = clamp(p.xyz, -1.0, 1.0) * 2.0 - p.xyz;  // box fold: min3, max3, mad3
		float r2 = dot(p.xyz, p.xyz);  // dp3
        if (r2 > 5.) {
            return i;
        }
		p.xyzw *= clamp(max(MR2/r2, MR2), 0.0, 1.0);  // sphere fold: div1, max1.sat, mul4
		p.xyzw = p*mbox_scalevec + p0;  // mad4
	}
	float d = (length(p.xyz) - mbox_C1) / p.w - mbox_C2;
    return 0;
}

float mandelbox(vec3 position, int iters=7) {
    float mbox_C2 = pow(abs(SCALE), float(1-iters));
	// distance estimate
	vec4 p = vec4(position.xyz, 1.0), p0 = vec4(position.xyz, 1.0);  // p.w is knighty's DEfactor
	for (int i=0; i<iters; i++) {
		p.xyz = clamp(p.xyz, -1.0, 1.0) * 2.0 - p.xyz;  // box fold: min3, max3, mad3
		float r2 = dot(p.xyz, p.xyz);  // dp3
		p.xyzw *= clamp(max(MR2/r2, MR2), 0.0, 1.0);  // sphere fold: div1, max1.sat, mul4
		p.xyzw = p*mbox_scalevec + p0;  // mad4
	}
	float d = (length(p.xyz) - mbox_C1) / p.w - mbox_C2;
    return d;
}

float scene_old(vec3 p, out int material, float pixelConeSize=1.) {
	material = MATERIAL_OTHER;
	return mandelbox(p, 9);
}

float scene(vec3 p, out int material, float pixelConeSize=1.) {
	material = MATERIAL_OTHER;
    float d = length(p + 0.25*vec3(cos(secs*1.), 0., 0.)) - 1.;
    return d;
}

vec3 evalnormal(vec3 p) {
	vec2 e=vec2(1e-3, 0.f); //vec2 e=vec2(1e-5, 0.f);
	int m;
	return normalize(vec3(
				scene(p + e.xyy,m) - scene(p - e.xyy,m),
				scene(p + e.yxy,m) - scene(p - e.yxy,m),
				scene(p + e.yyx,m) - scene(p - e.yyx,m)
				));
}

void getCameraProjection(CameraParams cam, vec2 uv, out vec3 outPos, out vec3 outDir) {
	outPos = cam.pos + cam.dir + (uv.x - 0.5) * cam.right + (uv.y - 0.5) * cam.up;
	outDir = normalize(outPos - cam.pos);
}

vec2 shadowMarch(inout vec3 p, vec3 rd, int num_iters, float w, float mint, float maxt) {
    vec3 ro = p;
    int i;
    float omega = 1.3;
    float t = mint;
    int mat;
    float last_d = 0.;
    float step = 0.;
    float closest = MAX_DISTANCE;

    for (i = 0; i < num_iters; i++) {
        float d = scene(ro + t * rd, mat);

        bool sorFail = omega > 1. && (d + last_d) < step;
        if (sorFail) {
            step -= omega * step;
            omega = 1.;
        } else {
            step = d * omega;
        }

        //closest = min(closest, d);
        closest = min(closest, 0.5+0.5*d/(w*t) );

        last_d = d;

        if (d < 1e-5) {
            break;
        }

        if (t >= maxt) {
            break;
        }

        t += step;
    }

    p = ro + t * rd;
    closest = max(0., closest);
    //return vec2(t, closest*closest*(3.-2.*closest));
    return vec2(t, closest);
}

float depth_march(inout vec3 p, vec3 rd, out int material, out vec2 restart, int num_iters, out int out_iters, float maxDist=20.) {
    vec3 ro = p;
    int i;
    float t = 0.;
    int mat;
    float last_t = t;
    material = MATERIAL_OTHER;

    for (i = 0; i < num_iters; i++) {
        float d = scene(ro + t * rd, mat);

        float coneWorldRadius = PIXEL_RADIUS * ((t + PROJECTION_PLANE_DIST) / PROJECTION_PLANE_DIST);

        if (d <= coneWorldRadius) {
            // In depth rays we write the earlier, "safe", z value to the buffer.
            t = last_t;
            break;
        }

        last_t = t;
        t += d;

        if (t >= maxDist) {
            break;
        }
    }

    out_iters = i;

    if (t >= maxDist) {
        material = MATERIAL_SKY;
        return MAX_DISTANCE;
    }

    return t;
}

float march(inout vec3 p, vec3 rd, out int material, out vec2 restart, int num_iters, out int out_iters, float maxDist=20.) {
    vec3 ro = p;
    int i;
    float t = 0.;
    int mat;
    material = MATERIAL_OTHER;

    for (i = 0; i < num_iters; i++) {
        float d = scene(ro + t * rd, mat);

        float coneWorldRadius = PIXEL_RADIUS + (t * PIXEL_RADIUS) / NEAR_PLANE;

        if (d <= coneWorldRadius) {
            // In depth rays we write the earlier, "safe", z value to the buffer.
            material = mandelbox_material(ro + t * rd);
            break;
        }

        t += d;

        if (t >= maxDist) {
            break;
        }
    }

    out_iters = i;

    if (t >= maxDist) {
        material = MATERIAL_SKY;
        return MAX_DISTANCE;
    }

    return t;
}

// Raymarching loop based on techniques of Keinert et al. "Enhanced Sphere Tracing", 2014.
float march_old(inout vec3 p, vec3 rd, out int material, out vec2 restart, int num_iters, out int out_iters, float maxDist=20.) {
    vec3 ro = p;
    int i;
    float omega = 1.3;
    float t = 0.;
    float restart_t = t;
    float restart_error = 0.;
    float candidate_error = 1e9;
    float candidate_t = t;
    int mat;
    float last_d = 0.;
    float step = 0.;
    material = MATERIAL_OTHER;

    for (i = 0; i < num_iters; i++) {
        float d = scene(ro + t * rd, mat);

        bool sorFail = omega > 1. && (d + last_d) < step;
        if (sorFail) {
            step -= omega * step;
            omega = 1.;
        } else {
            step = d * omega;
        }

        float error = d / t; // FIXME: should be d/(t+1)? division of zero here

        if (d > last_d && error < restart_error) {
            restart_t = t;
            restart_error = error;
        }

        last_d = d;

        if (!sorFail && error < candidate_error) {
            candidate_t = t;
            candidate_error = error;
        }

        //if (!sorFail && error < 4.*PIXEL_RADIUS || t >= maxDist) {
        if (!sorFail && error < PIXEL_RADIUS || t >= maxDist) {
            material = mandelbox_material(ro + t * rd);
            //material = mat;
            break;
        }

        t += step;
    }

    restart = vec2(0, PIXEL_RADIUS);
    out_iters = i;

    if (t >= maxDist) {
        material = MATERIAL_SKY;
        t = MAX_DISTANCE;
        return t;
    }

    // Write out sky color if snapping the hit point to nearest geometry would introduce too much screen space error.
    if (i == num_iters && candidate_error > PIXEL_RADIUS * SNAP_INFLATE_FACTOR) {
        material = MATERIAL_SKY;
        return t;
    }

    restart = vec2(restart_t, restart_error);
    t = candidate_t;
    p = ro + t * rd;

    // See "Enhanced Sphere Tracing" section 3.4. and
    // section 3.1.1 in "Efficient Antialiased Rendering of 3-D Linear Fractals"
    for (int i = 0; i < 2; i++) {
        int temp;
        float e = t * 2. * PIXEL_RADIUS;
        t += scene(ro + t*rd, temp) - e;
    }

    return t;
}

float sampleAO(vec3 ro, vec3 rd)
{
    const int STEPS = 20;
    const float step = 0.05;
    float t = step;
    int mat=0;
    float obscurance = 0.;

    for (int i=0; i < STEPS; i++) {
        float d = scene(ro + t * rd, mat);
        obscurance += max(0., t - d) / t;
        t += step;
    }

    return 1. - obscurance / STEPS;
}

// Maps a ray index "i" into a bin index.
int tobin(int i)
{
    return findMSB(3*i+1)>>1;
}

// Maps a bin index into a starting ray index. Inverse of "tobin(i)."
int binto(int b)
{
    // Computes (4**b - 1) / 3
    // FIXME: replace with a lookup table
    int product = 1;
    for (int i = 0; i < b; i++)
        product *= 4;
    return (product - 1) / 3;
}

uint z2x_1(uint x)
{
    x = x & 0x55555555;
    x = (x | (x >> 1)) & 0x33333333;
    x = (x | (x >> 2)) & 0x0F0F0F0F;
    x = (x | (x >> 4)) & 0x00FF00FF;
    x = (x | (x >> 8)) & 0x0000FFFF;
    return x;
}

// Maps 32-bit Z-order index into 16-bit (x, y)
uvec2 z2xy(uint z)
{
    return uvec2(z2x_1(z), z2x_1(z>>1));
}

vec2 i2ray(int i, out ivec2 squareCoord, out int parentIdx, out int sideLength)
{
    int b = tobin(i);
    int start = binto(b);
    int z = i - start;
    uvec2 coord = z2xy(uint(z));
    int dim = 1 << b;
    int size = dim * dim;

    int parent_size = size >> 2; // (2 << (b-1));
    int parent = int(start - parent_size) + (z/4);

    float margin = 1.0 / float(2*dim);
    float step = (1.0 - 1. / float(dim)) / float(dim);

    squareCoord = ivec2(coord + vec2(.5));
    parentIdx = parent;
    sideLength = dim;

    vec2 uv = vec2(margin) + step * vec2(coord);

    if (i == 599) {
        debug_i = i;
        debug_parent = parent;
        debug_size = dim;
        debug_b = b;
        debug_start = start;
        debug_parent_size = parent_size;
    }

     return uv;
    //return step * vec2(coord);
}


void main() {
    ivec2 res = imageSize(zbuffer).xy;

    const int maxRayIndex = res.x * res.y;

    while (nextRayIndex < jumpBufferMaxElements) {
        int myIdx = atomicAdd(nextRayIndex, 1);
        if (myIdx >= jumpBufferMaxElements)
            return;

        int parentIdx = -2, sideLength = -1;
        ivec2 squareCoord;
        vec2 squareUV = i2ray(myIdx, squareCoord, parentIdx, sideLength);

        ivec2 pixelCoord = squareCoord; //ivec2(myIdx % res.x, myIdx / res.x);

        srand(frame, uint(pixelCoord.x), uint(pixelCoord.y));
        jenkins_mix();
        jenkins_mix();

        vec2 uv = vec2(pixelCoord) / vec2(res);
        uv = squareUV; // FIXME: requires division by 1023 in splatter to avoid scaling! why

        if (any(greaterThan(uv, vec2(1.0)))) {
            continue;
        }

        // Allow sampling half a pixel outside the screen
        if (any(lessThan(uv, vec2(-0.5/res.x, -0.5/res.y)))) {
            continue;
        }

        float parentDepth = jumps[parentIdx];

        CameraParams cam = cameras[1];
        vec3 p, dir;
        getCameraProjection(cam, uv, p, dir);

        //PIXEL_RADIUS = .5 * length(cameras[1].right) / res.x;
        PIXEL_RADIUS = sqrt(2.) * .5 * length(cam.right) / float(sideLength);
        PROJECTION_PLANE_DIST = length(p - cam.pos);
        NEAR_PLANE = length(cam.dir);

        int hitmat = MATERIAL_SKY;
        vec2 restart;
        int iters=0;
        float zdepth;

        p += dir * parentDepth;

        bool isLowestLevel = sideLength >= max(res.x, res.y);

        if (isLowestLevel) {
            zdepth = march(p, dir, hitmat, restart, 400, iters);
        } else {
            zdepth = depth_march(p, dir, hitmat, restart, 100, iters);
        }

        if (myIdx == 0) {
            debug_pixelRadius = PIXEL_RADIUS;
            debug_zdepth = zdepth;
            debug_parentDepth = parentDepth;
        }

        // if (sideLength >= 256) {
        //     zdepth = parentDepth;
        // }

        jumps[myIdx] = zdepth + 1e-6;

        if (!isLowestLevel) {
            continue;
        }

        float distance = length(p - cam.pos);

        vec3 color;
        vec3 skyColor = vec3(0., 0.2*abs(dir.y), 0.5 - dir.y);

        if (hitmat == MATERIAL_SKY) {
            color = skyColor;
        } else {
            vec3 normal = evalnormal(p);
            vec3 to_camera = normalize(cam.pos - p);
            vec3 to_light = normalize(vec3(-0.5, -1.0, 0.7));

            // HACK: Disable shading
            if (false) {
                vec3 shadowRayPos = p + to_camera * 1e-4;
                const float maxShadowDist = 10.;
                vec2 shadowResult = shadowMarch(shadowRayPos, to_light, 400, 9e-2, 0.01, maxShadowDist);
                float sun = min(shadowResult.x, maxShadowDist) / maxShadowDist;
                //sun = max(0., shadowResult.y - 0.1);
                //sun = (sun+0.1) * shadowResult.y;

                //float sun = min(1.0, shadowResult.y*1e3);
                sun = pow(sun, 2.);

                float ambient = sampleAO(p, normal);
                ambient = pow(ambient, 2.0);

                float facing = max(0., dot(normal, to_light));

                vec3 base = vec3(.5) + .5*vec3(sin(hitmat + vec3(0., .5, 1.)));


                //color = base * sun * vec3(facing);
                color = vec3(sun);
                vec3 skycolor = vec3(0.5, 0.7, 1.0);
                vec3 suncolor = vec3(1., 0.8, 0.5);
                color = base * (ambient * skycolor + facing * sun * suncolor);
                color = clamp(color, vec3(0.), vec3(10.));
                //color = vec3(0.5)+.5*cos( 10*vec3(iters)/600.  + vec3(0., 0.5, 1.));
            } else {
                color = vec3(normal); // * vec3(0., 1., 0.);
            }
        }

        color = vec3(pow(zdepth/10., 4.));
        //color = vec3(.0)+.5*sin(vec3(zdepth*40.) + vec3(0., 0.5, 1.));
        if (zdepth == 0.){
            color = vec3(1., 0., 0.);
        }
        if (iters == 1) {
            color = vec3(0., 0., 1.);
        }
        if (parentIdx < 0) {
            //color = vec3(0., 1., 0.);
        }

        if (true /* HACK: always write points */ || hitmat != MATERIAL_SKY) {
            int myPointOffset = atomicAdd(currentWriteOffset, 1);
            myPointOffset %= pointBufferMaxElements;

            points[myPointOffset].xyzw = vec4(p, 0.);
            points[myPointOffset].rgba = vec4(color, 1.);
        }

        imageStore(zbuffer, pixelCoord, vec4(zdepth));
        if (zdepth >= 1e9) {
            for (int y=-1;y<=1;y++) {
                for (int x=-1;x<=1;x++) {
                    imageStore(edgebuffer, pixelCoord + ivec2(x, y), vec4(1.));
                }
            }
        }
    }
}

