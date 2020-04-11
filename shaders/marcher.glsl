#version 460
// the local thread block size; the program will be ran in sets of 16 by 16 threads.
layout(local_size_x = 16, local_size_y = 16) in;

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

uniform int source;
uniform int frame;
uniform float secs;
uniform float sceneID;
uniform ivec2 screenSize;
uniform vec2 screenBoundary;
uniform vec2 cameraJitter;
uniform vec3 sunDirection;
uniform vec3 sunColor;
uniform vec3 fogColor;
uniform vec3 fogScatterColor;
uniform samplerCube skybox;
uniform sampler2D skyIrradiance;

layout(r32f) uniform image2D zbuffer;
layout(r8) uniform image2D edgebuffer;
layout(rgba16f) uniform image2DArray samplebuffer;
layout(rg8) uniform image2DArray jitterbuffer;

struct CameraParams {
    vec3 pos;
    float padding0;
    vec3 dir;
    float nearplane;
    vec3 up;
    float aspect;
    vec3 right;
    float padding3;
};

struct RgbPoint {
    vec3 xyz;
    uint normalSpecularSun;
    vec3 rgba;
    float sec;
};

uniform int pointBufferMaxElements;
//uniform int jumpBufferMaxElements;
uniform int rayIndexBufferMaxElements;

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

layout(std430) buffer rayIndexBuffer {
    int rayIndices[];
};

layout(std430) buffer uvBuffer {
    vec2 debug_uvs[];
};

layout(std430) buffer radiusBuffer {
    float radiuses[];
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
    float debug_parent_t;
    float debug_child_t;
    float debug_nearPlane;
    float debug_projPlaneDist;
    float debug_parentUVx;
    float debug_parentUVy;
    float debug_childUVx;
    float debug_childUVy;
};

layout(std430) buffer stepBuffer {
    float debug_steps[];
};

#define USE_ANALYTIC_CONE_STEP 1
#define USE_HIT_REFINEMENT 0
#define USE_TREE 1

// This factor how many pixel radiuses of screen space error do we allow
// in the "near geometry snapping" at the end of "march" loop. Without it
// the skybox color leaks through with low grazing angles.
const float SNAP_INFLATE_FACTOR = 3.;

// Ray's maximum travel distance.
const float MAX_DISTANCE = 1e9;


float PIXEL_RADIUS;
float PROJECTION_PLANE_DIST;
float NEAR_PLANE;
float STEP_FACTOR;

const int PARENT_INDEX  = 16;
const int CHILD_INDEX = 112;
int globalMyIdx;

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

float scene(vec3 p, out int material, float pixelConeSize=1.) {
	material = MATERIAL_OTHER;
	return mandelbox(p, 9);
}

float scene_ball(vec3 p, out int material, float pixelConeSize=1.) {
	material = MATERIAL_OTHER;
    float d = length(p - vec3(0., -4., 0.)) - 2.;
    return d;
}

void getCameraProjection(CameraParams cam, vec2 uv, out vec3 outPos, out vec3 outDir) {
	outPos = cam.pos + cam.dir + (uv.x - 0.5) * cam.right + (uv.y - 0.5) * cam.up;
	outDir = normalize(outPos - cam.pos);
}

float depth_march(inout vec3 p, vec3 rd, out int material, out vec2 restart, int num_iters, out int out_iters, float start_t=0., float end_t=20.) {
    vec3 ro = p;
    int i;
    float t = start_t;
    int mat;
    float last_t = t;
    material = MATERIAL_OTHER;

    for (i = 0; i < num_iters; i++) {
        float d = scene(ro + t * rd, mat);
        float coneWorldRadius = PIXEL_RADIUS * (t+PROJECTION_PLANE_DIST) / PROJECTION_PLANE_DIST;

        if (globalMyIdx == PARENT_INDEX) {
            debug_steps[4*i] = t;
            debug_steps[4*i + 1] = d;
            debug_steps[4*i + 2] = coneWorldRadius;
            debug_steps[4*i + 3] = PIXEL_RADIUS;
        }

        if (globalMyIdx == CHILD_INDEX) {
            debug_steps[1000*4 + 4*i] = t;
            debug_steps[1000*4 + 4*i + 1] = d;
            debug_steps[1000*4 + 4*i + 2] = coneWorldRadius;
            debug_steps[1000*4 + 4*i + 3] = PIXEL_RADIUS;
        }

        if (d <= coneWorldRadius) {
            // In depth rays we write the earlier, "safe", z value to the buffer.
            t = last_t;
            break;
        }

        last_t = (t + d) - coneWorldRadius;

        #if USE_ANALYTIC_CONE_STEP
        t = (t + d) * STEP_FACTOR;
        #else
        t = t + d;
        #endif

        if (t >= end_t) {
            break;
        }
    }

    out_iters = i;

    if (t >= end_t) {
        material = MATERIAL_SKY;
        t = MAX_DISTANCE;
    }

    if (globalMyIdx == PARENT_INDEX) {
        debug_parent_t = t;
    }

    if (globalMyIdx == CHILD_INDEX) {
        debug_child_t = t;
    }


    return t;
}

// Raymarching loop based on techniques of Keinert et al. "Enhanced Sphere Tracing", 2014.
float march(inout vec3 p, vec3 rd, out int material, out vec2 restart, int num_iters, out int out_iters, float start_t=0., float end_t=20.) {
    vec3 ro = p;
    int i;
    float omega = 1.3;
    float t = start_t;
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

        // Worst case distance to surface in screen space.
        float error = d / (t + PROJECTION_PLANE_DIST);

        if (d > last_d && error < restart_error) {
            restart_t = t;
            restart_error = error;
        }

        last_d = d;

        if (!sorFail && error < candidate_error) {
            candidate_t = t;
            candidate_error = error;
        }

        if (!sorFail && error < 1. * PIXEL_RADIUS || t >= end_t) {
            material = mandelbox_material(ro + t * rd);
            break;
        }

        t += step;
    }

    restart = vec2(0, PIXEL_RADIUS);
    out_iters = i;

    if (t >= end_t) {
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

#if USE_HIT_REFINEMENT
    // See "Enhanced Sphere Tracing" section 3.4. and
    // section 3.1.1 in "Efficient Antialiased Rendering of 3-D Linear Fractals"
    for (int i = 0; i < 2; i++) {
        int temp;
        float e = t * 2. * PIXEL_RADIUS;
        t += scene(ro + t*rd, temp) - e;
    }
#endif

    return t;
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

vec3 evalnormal_rough(vec3 p) {
    vec2 e=vec2(1e-3, 0.f);
    int m;
    return normalize(vec3(
                scene(p + e.xyy,m) - scene(p - e.xyy,m),
                scene(p + e.yxy,m) - scene(p - e.yxy,m),
                scene(p + e.yyx,m) - scene(p - e.yyx,m)
                ));
}

vec2 shadowMarch(in vec3 p, vec3 rd, int num_iters, float w, float mint, float maxt) {
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
float sampleAO(vec3 ro, vec3 rd)
{
    const int STEPS = 10;
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

uvec2 i2gridCoord(int i, out int idim) {
    int b = tobin(i);
    int start = binto(b);
    int z = i - start;
    idim = 1 << b;
    return z2xy(uint(z));
}

vec2 octWrap( vec2 v )
{
    //return ( 1.0 - abs( v.yx ) ) * ( v.xy >= 0.0 ? 1.0 : -1.0 );

    //vec2 v2 = vec2(
    //    v.x >= 0. ? 1. : -1.,
    //    v.y >= 0. ? 1. : -1.);
    vec2 v2 = mix(vec2(-1.), vec2(1.), greaterThanEqual(v.xy, vec2(0.)));
    return ( 1.0 - abs( v.yx ) ) * v2;
}

vec2 encodeNormal( vec3 n )
{
    n /= ( abs( n.x ) + abs( n.y ) + abs( n.z ) );
    //n.xy = n.z >= 0.0 ? n.xy : OctWrap( n.xy );
    // vec2 nw = OctWrap( n.xy );
    // n.x = n.z >= 0.0 ? n.x : nw.x;
    // n.y = n.z >= 0.0 ? n.y : nw.y;

    n.xy = mix(octWrap( n.xy ), n.xy, bvec2(n.z >= 0.));
    n.xy = n.xy * 0.5 + vec2(0.5);
    return n.xy;
}

vec3 decodeNormal( vec2 f )
{
    f = f * 2.0 - vec2(1.0);

    // https://twitter.com/Stubbesaurus/status/937994790553227264
    vec3 n = vec3( f.x, f.y, 1.0 - abs( f.x ) - abs( f.y ) );
    float t = clamp( -n.z, 0., 1. );
    //n.xy += n.xy >= 0.0 ? -t : t;
    n.xy += mix(vec2(t), vec2(-t), greaterThanEqual(n.xy, vec2(0.)));
    //n.x += n.x >= 0.0 ? -t : t;
    //n.y += n.y >= 0.0 ? -t : t;
    return normalize( n );
}

// https://learnopengl.com/PBR/IBL/Diffuse-irradiance
const vec2 invAtan = vec2(0.1591, 0.3183);
vec2 dirToSpherical(vec3 direction)
{
    vec2 uv = vec2(atan(direction.z, direction.x), asin(direction.y));
    uv *= invAtan;
    uv += 0.5;
    return uv;
}

vec3 sampleSky(vec3 dir) {
    vec3 img = texture(skyIrradiance, dirToSpherical(vec3(dir.x, -dir.y, dir.z))).rgb;
    vec3 skyColor = vec3(0., 0.2*abs(dir.y), 0.5 - dir.y);
    return img.bgr;
}



vec2 i2ray(int i, out ivec2 squareCoord, out int parentIdx, out int sideLength)
{
    int b = tobin(i);
    int start = binto(b);
    int z = i - start;
    uvec2 coord = z2xy(uint(z));
    int idim = 1 << b;
    int size = idim * idim;
    float dim = float(idim);

    int parent_size = size / 4;
    int parent = int(start - parent_size) + (z/4);

    squareCoord = ivec2(coord + vec2(.5));
    parentIdx = parent;
    sideLength = idim;

    vec2 uv = vec2(0.5/dim) + coord / vec2(dim);


    if (i == 599) {
        debug_i = i;
        debug_parent = parent;
        debug_size = idim;
        debug_b = b;
        debug_start = start;
        debug_parent_size = parent_size;
    }

     return uv;
}

void main() {
    ivec2 res = screenSize;

    const int maxRayIndex = res.x * res.y;

    while (nextRayIndex < rayIndexBufferMaxElements) {
        int arrayIdx = atomicAdd(nextRayIndex, 1);
        CameraParams cam = cameras[1];

#if USE_TREE
        if (arrayIdx >= rayIndexBufferMaxElements)
            return;
        int myIdx = rayIndices[arrayIdx];
        //int myIdx = arrayIdx;
        globalMyIdx = myIdx;
        int idim;
        int parentIdx = -2, sideLength = -1;

        ivec2 squareCoord;
        vec2 squareUV = i2ray(myIdx, squareCoord, parentIdx, sideLength);

        if (squareUV.x > screenBoundary.x || squareUV.y > screenBoundary.y) {
            continue;
        }
        squareUV /= screenBoundary.xx;

        ivec2 pixelCoord = ivec2(squareUV * vec2(1., cam.aspect) * res.xy);
#else
        int myIdx = arrayIdx;
        if (myIdx >= res.x * res.y)
            continue;
        int parentIdx = 0;
        int sideLength = res.x;
        ivec2 pixelCoord = ivec2(myIdx % res.x, myIdx / res.x);
        vec2 squareUV = (vec2(0.5) / vec2(sideLength)) + pixelCoord / vec2(sideLength);
#endif


        srand(frame, uint(pixelCoord.x), uint(pixelCoord.y));
        jenkins_mix();
        jenkins_mix();

        vec2 uv = squareUV;

        /*
        if (any(greaterThan(uv, vec2(1.0)))) {
            continue;
        }

        // Allow sampling half a pixel outside the screen
        if (any(lessThan(uv, vec2(-0.5/res.x, -0.5/res.y)))) {
            continue;
        }
        */

#if USE_TREE
        float parentDepth = jumps[parentIdx];
        //parentDepth = 0.; // HACK!!!
        if (parentDepth >= MAX_DISTANCE) {
            jumps[myIdx] = parentDepth;
            continue;
        }
#else
        float parentDepth = 0.;
#endif

        bool isLowestLevel = sideLength >= max(res.x, res.y);

        if (isLowestLevel) {
            vec2 jitter = vec2(rand() - 0.5, rand() - 0.5);
            jitter /= res;
            jitter *= 0.9;
            uv += jitter;
        }

        vec3 p, dir;
        getCameraProjection(cam, uv, p, dir);

        vec3 rp = p - cam.pos;
        PROJECTION_PLANE_DIST = length(rp);
        NEAR_PLANE = length(cam.dir);
        PIXEL_RADIUS = .5 * sqrt(2.) * length(cam.right) / float(sideLength);

#if USE_ANALYTIC_CONE_STEP
        {
            float aperture = 2. * PIXEL_RADIUS;
            float C = sqrt(aperture * aperture + 1.);
            STEP_FACTOR = C / (C - aperture);
        }
#endif

        int hitmat = MATERIAL_SKY;
        vec2 restart;
        int iters=0;
        float zdepth = -1.;

        if (isLowestLevel) {
            zdepth = march(p, dir, hitmat, restart, 400, iters, parentDepth);
        } else {
            zdepth = depth_march(p, dir, hitmat, restart, 400, iters, parentDepth);
        }

        if (myIdx == CHILD_INDEX) {
            debug_pixelRadius = iters;
            debug_nearPlane = NEAR_PLANE;
            debug_projPlaneDist = PROJECTION_PLANE_DIST;
            debug_zdepth = zdepth;
            debug_parentDepth = parentDepth;
            debug_childUVx = uv.x;
            debug_childUVy = uv.y;
        }

        if (myIdx == PARENT_INDEX) {
            debug_parentUVx = uv.x;
            debug_parentUVy = uv.y;
        }

#if USE_TREE
        jumps[myIdx] = zdepth;
#endif
        //radiuses[myIdx] = (PIXEL_RADIUS / (2.* length(cam.right)));
        radiuses[myIdx] = PIXEL_RADIUS;
        debug_uvs[myIdx] = uv;

        if (!isLowestLevel) {
            continue;
        }

        vec3 color;

        if (hitmat == MATERIAL_SKY) {
            color = vec3(0.);
        } else {
            vec3 normal = evalnormal(p);
            vec3 roughNormal = evalnormal_rough(p);
            vec3 to_camera = normalize(cam.pos - p);
            vec3 to_light = sunDirection;
            //to_light = normalize(vec3(-0.4, 1., 0.0));

            vec3 shadowRayPos = p + to_camera * 1e-4;
            const float maxShadowDist = 30.;
            vec2 shadowResult = shadowMarch(shadowRayPos, to_light, 200, 9e-2, 5e-3, maxShadowDist);
            float sun = min(shadowResult.x, maxShadowDist) / maxShadowDist;
            //sun = pow(shadowResult.y-0.1, 3.0);

            sun = pow(sun, 2.);
            sun *= 4.;

            float ambient = max(0., sampleAO(p, normal));
            ambient = pow(ambient, 1.3);

            float facing = max(0., dot(normal, to_light));

            vec3 base = vec3(.5) + .5*vec3(sin(hitmat + vec3(0., .5, 1.)));
            float shininess = mod(hitmat * 0.2 + 0.5, 1.0);
            base = pow(base, vec3(1.3));
            //shininess = pow(shininess, 1.);
            if (hitmat == 2) {
                //shininess = 1.0;
            }
            vec3 suncol = sunColor;

            //color = base * sun * vec3(facing);
            //vec3 skycolor = mix(vec3(1.), fogColor, .8) * sampleSky(roughNormal);
            vec3 skycolor = sampleSky(roughNormal);
            if (suncol.g > suncol.r) {
                skycolor = suncol * 0.5;
            }
            color = base * (ambient * skycolor + facing * sun * suncol);
            //color=vec3(ambient);
            color = clamp(color, vec3(0.), vec3(2.));
            //shininess = 1.;

            //shininess = 0.;
            //color *= 0.5;
            //color *= 0.;

            //color = vec3(sun);
            //color = vec3(0.5)+.5*cos( 10*vec3(iters)/600.  + vec3(0., 0.5, 1.));

            int myPointOffset = atomicAdd(currentWriteOffset, 1);
            myPointOffset %= pointBufferMaxElements;

            vec2 packedNormal = encodeNormal(roughNormal);

            points[myPointOffset].xyz = p;
            points[myPointOffset].normalSpecularSun = packUnorm4x8(vec4(packedNormal, shininess, sun));
            points[myPointOffset].rgba = vec3(color);
            points[myPointOffset].sec = sceneID;
        }

        //color = vec3(uv, pow(zdepth/10., 5.));
        //color = vec3(1.);


        imageStore(zbuffer, pixelCoord, vec4(zdepth));
        if (zdepth >= MAX_DISTANCE) {
            for (int y=-1;y<=1;y++) {
                for (int x=-1;x<=1;x++) {
                    //float old = imageLoad(edgebuffer, pixelCoord + ivec2(x, y)).x;
                    //imageStore(edgebuffer, pixelCoord + ivec2(x, y), vec4(old+1./10.));
                    imageStore(edgebuffer, pixelCoord + ivec2(x, y), vec4(1));
                    //imageAtomicAdd(edgebuffer, pixelCoord + ivec2(x, y), 1.);
                }
            }
        }
    }
}

