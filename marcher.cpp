
#include "testbench.h"
#include <cinttypes>
#include <cassert>

struct CameraParameters {
	vec3 pos;
	float padding0;
	vec3 dir;
	float nearplane;
	vec3 up;
	float padding2;
	vec3 right;
	float padding3;
};

double getTime()
{
	static bool initialized;
	static LARGE_INTEGER StartingTime;
	static LARGE_INTEGER Frequency;

	if (!initialized) {
		QueryPerformanceFrequency(&Frequency);
		QueryPerformanceCounter(&StartingTime);
		initialized = true;
	}

	// Activity to be timed

	LARGE_INTEGER now, ElapsedMicroseconds;
	QueryPerformanceCounter(&now);
	ElapsedMicroseconds.QuadPart = now.QuadPart - StartingTime.QuadPart;
	return (double)ElapsedMicroseconds.QuadPart / 1000000.;
}

const int screenw = 1024, screenh = 1024;
static constexpr GLuint SAMPLE_BUFFER_TYPE = GL_RGBA16F;
static constexpr GLuint JITTER_BUFFER_TYPE = GL_RG8;

static void cameraPath(float t, CameraParameters& cam)
{
	float tt = t * 0.1f / 8.;
	//cam.pos = vec3(0.5f*sin(tt), 0.f, 6.f + 0.5f*cos(tt));
	cam.pos = vec3(0. + 2.0 * sin(tt), -4., 7.f + 0.1f*cos(tt));
	cam.dir = normalize(vec3(0.5f*cos(tt*0.5f), 0.3 + 0.2f*sin(tt), -1.f));
	//cam.pos = vec3(0., 0., 4.);
	//cam.dir = vec3(0., 0., -1.);
	cam.right = normalize(cross(cam.dir, vec3(0.f, 1.f, 0.f)));
	cam.up = cross(cam.dir, cam.right);
	
	float nearplane = 0.1f;
	float zoom = 1.0f;
	cam.dir *= nearplane;
	cam.right *= nearplane;
	cam.right /= zoom;
	cam.up *= nearplane;
	cam.up /= zoom;

	cam.nearplane = length(cam.dir);
}

static void setWrapToClamp(GLuint tex) {
	glTextureParameteri(tex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTextureParameteri(tex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

struct RgbPoint {
	vec4 xyzw;
	vec4 rgba;
};

int tobin(int i)
{
	return int(log2(3 * i + 1)) >> 1;
}

// Maps a bin index into a starting ray index. Inverse of "tobin(i)."
int binto(int b)
{
	// Computes (4**b - 1) / 3
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
	return uvec2(z2x_1(z), z2x_1(z >> 1));
}

// How many nodes must a full quadtree have when leaf layer has "dim" nodes.
int dim2nodecount(int dim)
{
	return binto(int(ceil(log2(dim))) + 1);
}

constexpr int MAX_POINT_COUNT = 1. * 1024 * 1024;

int main() {

	// create window and context. can also do fullscreen and non-visible window for compute only
	OpenGL context(screenw, screenh, "raymarcher");
	
	// load a font to draw text with -- any system font or local .ttf file should work
	Font font(L"Consolas");

	// shader variables; could also initialize them here, but it's often a good idea to
	// do that at the callsite (so input/output declarations are close to the bind code)
	Program march, edgeDetect, draw, sampleResolve, headerUpdate, pointSplat;

	std::wstring noisePaths[] = {
		L"assets/bluenoise/LDR_RGB1_0.png",
		L"assets/bluenoise/LDR_RGB1_1.png",
		L"assets/bluenoise/LDR_RGB1_2.png",
		L"assets/bluenoise/LDR_RGB1_3.png",
		L"assets/bluenoise/LDR_RGB1_4.png",
		L"assets/bluenoise/LDR_RGB1_5.png",
		L"assets/bluenoise/LDR_RGB1_6.png",
		L"assets/bluenoise/LDR_RGB1_7.png",
		L"assets/bluenoise/LDR_RGB1_8.png",
		L"assets/bluenoise/LDR_RGB1_9.png",
		L"assets/bluenoise/LDR_RGB1_10.png",
		L"assets/bluenoise/LDR_RGB1_11.png",
		L"assets/bluenoise/LDR_RGB1_12.png",
		L"assets/bluenoise/LDR_RGB1_13.png",
		L"assets/bluenoise/LDR_RGB1_14.png",
		L"assets/bluenoise/LDR_RGB1_15.png",
		L"assets/bluenoise/LDR_RGB1_16.png",
		L"assets/bluenoise/LDR_RGB1_17.png",
		L"assets/bluenoise/LDR_RGB1_18.png",
		L"assets/bluenoise/LDR_RGB1_19.png",
		L"assets/bluenoise/LDR_RGB1_20.png",
		L"assets/bluenoise/LDR_RGB1_21.png",
		L"assets/bluenoise/LDR_RGB1_22.png",
		L"assets/bluenoise/LDR_RGB1_23.png",
		L"assets/bluenoise/LDR_RGB1_24.png",
		L"assets/bluenoise/LDR_RGB1_25.png",
		L"assets/bluenoise/LDR_RGB1_26.png",
		L"assets/bluenoise/LDR_RGB1_27.png",
		L"assets/bluenoise/LDR_RGB1_28.png",
		L"assets/bluenoise/LDR_RGB1_29.png",
		L"assets/bluenoise/LDR_RGB1_30.png",
		L"assets/bluenoise/LDR_RGB1_31.png",
		L"assets/bluenoise/LDR_RGB1_32.png",
		L"assets/bluenoise/LDR_RGB1_33.png",
		L"assets/bluenoise/LDR_RGB1_34.png",
		L"assets/bluenoise/LDR_RGB1_35.png",
		L"assets/bluenoise/LDR_RGB1_36.png",
		L"assets/bluenoise/LDR_RGB1_37.png",
		L"assets/bluenoise/LDR_RGB1_38.png",
		L"assets/bluenoise/LDR_RGB1_39.png",
		L"assets/bluenoise/LDR_RGB1_40.png",
		L"assets/bluenoise/LDR_RGB1_41.png",
		L"assets/bluenoise/LDR_RGB1_42.png",
		L"assets/bluenoise/LDR_RGB1_43.png",
		L"assets/bluenoise/LDR_RGB1_44.png",
		L"assets/bluenoise/LDR_RGB1_45.png",
		L"assets/bluenoise/LDR_RGB1_46.png",
		L"assets/bluenoise/LDR_RGB1_47.png",
		L"assets/bluenoise/LDR_RGB1_48.png",
		L"assets/bluenoise/LDR_RGB1_49.png",
		L"assets/bluenoise/LDR_RGB1_50.png",
		L"assets/bluenoise/LDR_RGB1_51.png",
		L"assets/bluenoise/LDR_RGB1_52.png",
		L"assets/bluenoise/LDR_RGB1_53.png",
		L"assets/bluenoise/LDR_RGB1_54.png",
		L"assets/bluenoise/LDR_RGB1_55.png",
		L"assets/bluenoise/LDR_RGB1_56.png",
		L"assets/bluenoise/LDR_RGB1_57.png",
		L"assets/bluenoise/LDR_RGB1_58.png",
		L"assets/bluenoise/LDR_RGB1_59.png",
		L"assets/bluenoise/LDR_RGB1_60.png",
		L"assets/bluenoise/LDR_RGB1_61.png",
		L"assets/bluenoise/LDR_RGB1_62.png",
		L"assets/bluenoise/LDR_RGB1_63.png",

	};
	Texture<GL_TEXTURE_2D_ARRAY> noiseTextures = loadImageArray(noisePaths, sizeof(noisePaths)/sizeof(std::wstring));

	Texture<GL_TEXTURE_2D_ARRAY> abuffer;
	Texture<GL_TEXTURE_2D> gbuffer;
	Texture<GL_TEXTURE_2D> zbuffer;
	Texture<GL_TEXTURE_2D> edgebuffer;
	Texture<GL_TEXTURE_2D_ARRAY> samplebuffer;
	Texture<GL_TEXTURE_2D_ARRAY> jitterbuffer;
	Buffer cameraData;
	Buffer pointBufferHeader;
	Buffer pointBuffer;
	Buffer colorBuffer, sampleWeightBuffer;
	Buffer jumpbuffer;
	Buffer uvbuffer;
	Buffer radiusbuffer;
	Buffer debugBuffer;
	Buffer stepBuffer;

	//Framebuffer fbo;
	//glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	setWrapToClamp(abuffer);
	setWrapToClamp(gbuffer);
	setWrapToClamp(zbuffer);
	setWrapToClamp(samplebuffer);

	int renderw = screenw, renderh = screenh;

	glTextureStorage3D(abuffer, 1, GL_RGBA32F, screenw, screenh, 2);
	glTextureStorage2D(gbuffer, 1, GL_RGBA32F, renderw, renderh);
	glTextureStorage2D(edgebuffer, 1, GL_R8, renderw, renderh);
	glTextureStorage2D(zbuffer, 1, GL_R32F, renderw, renderh);
	glTextureStorage3D(samplebuffer, 1, SAMPLE_BUFFER_TYPE, renderw, renderh, 1);
	glTextureStorage3D(jitterbuffer, 1, JITTER_BUFFER_TYPE, renderw, renderh, 1);

	int pointsSplatted = 0;
	int frame = 0;
	int noiseLayer = -1;
	CameraParameters cameras[2] = {};
	glNamedBufferStorage(cameraData, sizeof(cameras), NULL, GL_DYNAMIC_STORAGE_BIT);
	glNamedBufferStorage(pointBufferHeader, 3 * sizeof(int), NULL, GL_DYNAMIC_STORAGE_BIT); // TODO read bit only for debugging
	glNamedBufferStorage(pointBuffer, sizeof(RgbPoint) * MAX_POINT_COUNT, NULL, GL_DYNAMIC_STORAGE_BIT); // TODO read bit only for debugging
	glNamedBufferStorage(colorBuffer, screenw * screenh * 3 * sizeof(int), NULL, 0);
	glNamedBufferStorage(sampleWeightBuffer, screenw * screenh * sizeof(int), NULL, 0);
	glNamedBufferStorage(debugBuffer, 1024 * sizeof(int), NULL, 0);
	glNamedBufferStorage(stepBuffer, 50000 * sizeof(float), NULL, 0);
	
	int jumpBufferMaxElements = dim2nodecount(max(screenw, screenh));
	glNamedBufferStorage(jumpbuffer, jumpBufferMaxElements * sizeof(float), NULL, 0);
	glNamedBufferStorage(radiusbuffer, jumpBufferMaxElements * sizeof(float), NULL, 0);
	glNamedBufferStorage(uvbuffer, 2 * jumpBufferMaxElements * sizeof(float), NULL, 0);

	int zero = 0;
	glClearNamedBufferData(pointBufferHeader, GL_R32I, GL_RED_INTEGER, GL_INT, &zero);
	glClearNamedBufferData(pointBuffer, GL_R32I, GL_RED_INTEGER, GL_INT, &zero);
	glClearNamedBufferData(debugBuffer, GL_R32I, GL_RED_INTEGER, GL_INT, &zero);
	float minusone = -1.f;
	glClearNamedBufferData(stepBuffer, GL_R32F, GL_RED, GL_FLOAT, &minusone);

	int thousand = 1000;
	int hundred = 100;
	glClearNamedBufferData(colorBuffer, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &thousand);
	glClearNamedBufferData(sampleWeightBuffer, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &hundred);

	int headerSize = -1;
	glGetNamedBufferParameteriv(pointBufferHeader, GL_BUFFER_SIZE, &headerSize);
	printf("pointBufferHeader size: %d bytes\n", headerSize);
	GLint64 pointBufferSize = -1;
	glGetNamedBufferParameteri64v(pointBuffer, GL_BUFFER_SIZE, &pointBufferSize);
	int pointBufferMaxElements = static_cast<int>(pointBufferSize / sizeof(RgbPoint));
	printf("pointBuffer size: %" PRId64 " bytes = %.3f MiB\n", pointBufferSize, pointBufferSize / 1024. / 1024.);
	printf("pointBufferMaxElements: %d\n", pointBufferMaxElements);

	GLint64 jumpBufferSize = -1;
	glGetNamedBufferParameteri64v(jumpbuffer, GL_BUFFER_SIZE, &jumpBufferSize);
	printf("jumpBuffer size: %" PRId64 " bytes = %.3f MiB\n", jumpBufferSize, jumpBufferSize / 1024. / 1024.);

	while (loop()) // loop() stops if esc pressed or window closed
	{
		// timestamp objects make gl queries at those locations; you can substract them to get the time
		TimeStamp start;
		float secs = getTime(); //fmod(frame / 60.f, 2.0) + 21.;
		secs = 0.;
		float futureInterval = 0. / 60.f;
		cameraPath(secs + futureInterval, cameras[1]);
		glNamedBufferSubData(cameraData, 0, sizeof(cameras), &cameras);
		float zeroFloat = 0.f;
		glClearNamedBufferData(jumpbuffer, GL_R32F, GL_RED, GL_FLOAT, &zeroFloat);

		{
			int layer;
			while ((layer = rand() % 64) == noiseLayer);
			noiseLayer = layer;
			//printf("noiseLayer: %d\n", noiseLayer);
		}

		glDisable(GL_BLEND);

		if (!march)
			march = createProgram("shaders/marcher.glsl");

		glUseProgram(march);

		glUniform1i("frame", frame);
		glUniform1f("secs", secs);
		glUniform1i("pointBufferMaxElements", pointBufferMaxElements);
		glUniform1i("jumpBufferMaxElements", jumpBufferMaxElements);
		bindBuffer("cameraArray", cameraData);
		bindBuffer("pointBufferHeader", pointBufferHeader);
		bindBuffer("pointBuffer", pointBuffer);
		bindBuffer("jumpBuffer", jumpbuffer);
		bindBuffer("uvBuffer", uvbuffer);
		bindBuffer("radiusBuffer", radiusbuffer);
		bindBuffer("debugBuffer", debugBuffer);
		bindBuffer("stepBuffer", stepBuffer);
		glUniform3i("noiseOffset", rand() % 64, rand() % 64, noiseLayer);
		bindTexture("noiseTextures", noiseTextures);
		//bindImage("gbuffer", 0, gbuffer, GL_WRITE_ONLY, GL_RGBA32F);
		bindImage("zbuffer", 0, zbuffer, GL_WRITE_ONLY, GL_R32F);
		bindImage("edgebuffer", 0, edgebuffer, GL_WRITE_ONLY, GL_R8);
		bindImage("samplebuffer", 0, samplebuffer, GL_WRITE_ONLY, SAMPLE_BUFFER_TYPE);
		bindImage("jitterbuffer", 0, jitterbuffer, GL_WRITE_ONLY, JITTER_BUFFER_TYPE);

		//glDispatchCompute((screenw+17)/16, (screenh+17)/16, 1); // round up and add extra context
		glDispatchCompute(screenw/16, screenh/16, 1); // TODO scale to max GPU occupancy?
		//glDispatchCompute(10, 20, 1); 

		// we're writing to an image in a shader, so we should have a barrier to ensure the writes finish
		// before the next shader call (wasn't an issue on my hardware in this case, but you should always make sure
		// to place the correct barriers when writing from compute shaders and reading in subsequent shaders)
		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT | GL_ATOMIC_COUNTER_BARRIER_BIT);

		TimeStamp drawTime;

		if (!headerUpdate) {
			headerUpdate = createProgram(
				GLSL(460,
					layout(local_size_x = 16, local_size_y = 16) in;

					layout(std430) buffer pointBufferHeader {
						int currentWriteOffset;
						int pointsSplatted;
						int nextRayIndex;
					};

					uniform int pointBufferMaxElements;

					void main() {
						currentWriteOffset = currentWriteOffset % pointBufferMaxElements;
						pointsSplatted = 0;
						nextRayIndex = 0;
					}
				)
			);
		}

		glUseProgram(headerUpdate);
		glUniform1i("pointBufferMaxElements", pointBufferMaxElements);
		bindBuffer("pointBufferHeader", pointBufferHeader);
		glDispatchCompute(1, 1, 1);
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

		if (!pointSplat) {
			pointSplat = createProgram(
				GLSL(460,
					layout(local_size_x = 128, local_size_y = 1) in;

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

			layout(r8) uniform image2D edgebuffer;
			uniform int pointBufferMaxElements;
			uniform int numberOfPointsToSplat;
			uniform ivec2 screenSize;
			uniform ivec3 noiseOffset;
			uniform sampler2DArray noiseTextures;

			vec3 getNoise(ivec2 coord)
			{
				return texelFetch(noiseTextures,
					ivec3((coord.x + noiseOffset.x) % 64, (coord.y + noiseOffset.y) % 64, noiseOffset.z),0).rgb;
			}

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

			layout(std430) buffer colorBuffer {
				uint colors[];
			};

			layout(std430) buffer sampleWeightBuffer {
				uint sampleWeights[];
			};

			layout(rgba32f) uniform image2D gbuffer;

			vec3 reprojectPoint(CameraParams cam, vec3 p) {
				vec3 op = p - cam.pos;
				float n = length(cam.dir);
				float z = dot(cam.dir, op) / n;
				vec3 pp = (op * n) / z;
				vec2 plane = vec2(
					dot(pp, cam.right) / dot(cam.right, cam.right),
					dot(pp, cam.up) / dot(cam.up, cam.up)
				);
				return vec3(plane + vec2(0.5), z);
			}

			void addRGB(uint pixelIdx, uvec3 c) {
				atomicAdd(colors[3 * pixelIdx + 0], c.r);
				atomicAdd(colors[3 * pixelIdx + 1], c.g);
				atomicAdd(colors[3 * pixelIdx + 2], c.b);
			}

			void main()
			{
				unsigned int invocationIdx =
					(gl_GlobalInvocationID.y * (gl_WorkGroupSize.x * gl_NumWorkGroups.x)
						+ gl_GlobalInvocationID.x);
				unsigned int baseIdx;

				if (invocationIdx >= pointBufferMaxElements)
					return;

				// We want to process "numberOfPointsToSplat" indices in a way that wraps around the buffer.
				if (currentWriteOffset >= numberOfPointsToSplat) {
					baseIdx = currentWriteOffset - numberOfPointsToSplat;
				}
				else {
					baseIdx = pointBufferMaxElements - (numberOfPointsToSplat - currentWriteOffset);
				};

				unsigned int index = (baseIdx + invocationIdx) % pointBufferMaxElements;
				vec4 pos = points[index].xyzw;
				vec4 color = points[index].rgba;
				vec3 c = color.rgb;

				// Raymarcher never produces pure (0, 0, 0) hits.
				if (pos == vec4(0.))
					return;

				vec3 camSpace = reprojectPoint(cameras[1], pos.xyz);
				vec2 screenSpace = camSpace.xy * vec2(screenSize.x, screenSize.y);
				int x = int(screenSpace.x);
				int y = int(screenSpace.y);

				if (x < 0 || y < 0 || x >= screenSize.x || y >= screenSize.y)
					return;

				int pixelIdx = screenSize.x * y + x;
				bool isEdge = imageLoad(edgebuffer, ivec2(x, y)).x > 0;
				float distance = length(pos.xyz - cameras[1].pos);
				float fog = pow(min(1., distance / 15.), 1.0);
				//c = mix(c, vec3(0.1, 0.1, 0.2)*0.1, fog);

				c = c / (vec3(1.) + c);
				c = clamp(c, vec3(0.), vec3(10.));

				float weight = max(0.1, min(1e3, 1. / (pow(camSpace.z / 3., 2.) + 0.001)));

				isEdge = false; // DEBUG: edge smoothing disabled for performance comparisons

				if (!isEdge) {
					uvec3 icolor = uvec3(weight * 8000 * c);
					addRGB(pixelIdx, icolor);
					atomicAdd(sampleWeights[pixelIdx], (uint(1000 * weight) << 16) | (255));
				} else {
					vec2 w = fract(screenSpace);
					vec4 ws = vec4(
						(1. - w.x) * (1. - w.y),
						w.x * (1. - w.y),
						(1. - w.x) * w.y,
						 w.x * w.y
					);

					int idx = screenSize.x * int(screenSpace.y) + int(screenSpace.x);

					// FIXME: don't write over image boundaries
					vec3 col = weight * c;
					addRGB(idx,					uvec3(8000 * ws[0] * col));
					addRGB(idx + 1,				uvec3(8000 * ws[1] * col));
					addRGB(idx + screenSize.x,		uvec3(8000 * ws[2] * col));
					addRGB(idx + screenSize.x + 1, uvec3(8000 * ws[3] * col));

					atomicAdd(sampleWeights[idx],						(uint(1000 * weight * ws[0]) << 16) | uint(255 * ws[0]));
					atomicAdd(sampleWeights[idx + 1],					(uint(1000 * weight * ws[1]) << 16) | uint(255 * ws[1]));
					atomicAdd(sampleWeights[idx + screenSize.x],		(uint(1000 * weight * ws[2]) << 16) | uint(255 * ws[2]));
					atomicAdd(sampleWeights[idx + screenSize.x + 1],	(uint(1000 * weight * ws[3]) << 16) | uint(255 * ws[3]));
				}

				atomicAdd(pointsSplatted, 1);
			}
			));
		}

		int numberOfPointsToSplat = MAX_POINT_COUNT;

		cameraPath(secs, cameras[1]);
		glNamedBufferSubData(cameraData, 0, sizeof(cameras), &cameras);

		glUseProgram(pointSplat);
		bindImage("gbuffer", 0, gbuffer, GL_READ_WRITE, GL_RGBA32F);
		glUniform1i("pointBufferMaxElements", pointBufferMaxElements);
		glUniform1i("numberOfPointsToSplat", numberOfPointsToSplat);
		int screenSize[] = { screenw, screenh };
		glUniform2i("screenSize", screenw, screenh);
		bindBuffer("pointBufferHeader", pointBufferHeader);
		bindBuffer("pointBuffer", pointBuffer);
		bindImage("edgebuffer", 0, edgebuffer, GL_READ_ONLY, GL_R8);
		bindBuffer("colorBuffer", colorBuffer);
		bindBuffer("sampleWeightBuffer", sampleWeightBuffer);
		glUniform3i("noiseOffset", rand() % 64, rand() % 64, noiseLayer);
		bindTexture("noiseTextures", noiseTextures);
		bindBuffer("cameraArray", cameraData);
		glDispatchCompute(numberOfPointsToSplat / 128 / 1, 1, 1);

		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT | GL_ATOMIC_COUNTER_BARRIER_BIT);

		TimeStamp splatTime;

		struct DebugData {
			int i;
			int parent;
			int size;
			int b;
			int start;
			int parent_size;
			float pixelRadius;
			float zdepth;
			float parentDepth;
			float parent_t;
			float child_t;
            float nearPlane;
            float projPlaneDist;
			float parentUVx;
			float parentUVy;
			float childUVx;
			float childUVy;
		};
		DebugData debugData = {};
		glGetNamedBufferSubData(debugBuffer, 0, sizeof(DebugData), &debugData);

		printf("debugData: i: %d, parent: %d, size: %d, b: %d, start: %d, parent_size: %d,\nradius: %f, zdepth: %f, parentDepth: %f\n",
			debugData.i, debugData.parent, debugData.size, debugData.b, debugData.start,
			debugData.parent_size,
			debugData.pixelRadius, debugData.zdepth, debugData.parentDepth);
		printf("nearPlane: %f, projPlaneDist: %f\n", debugData.nearPlane, debugData.projPlaneDist);
		printf("parent UV: (%f, %f), child UV: (%f, %f)\n",
			debugData.parentUVx, debugData.parentUVy, debugData.childUVx, debugData.childUVy);

		int data[2];
		glGetNamedBufferSubData(pointBufferHeader, 0, 8, data);
		//printf("currentWriteOffset: %d\n", data[0]);
		pointsSplatted = data[1];
		//printf("pointsSplatted: %d\t(%.3f million)\n", data[1], data[1]/1000000.);
		if (false) {
			GLint64 size = -1;
			glGetNamedBufferParameteri64v(stepBuffer, GL_BUFFER_SIZE, &size);
			std::vector<float> steps(size/sizeof(float), -1.f);
			glGetNamedBufferSubData(stepBuffer, 0, steps.size(), steps.data());
			int i = 0;
			while (steps[4*i] != -1.f) {
				printf("parent step [%d], t=%f\td=%f, cone: %f, pix: %f\n", i, steps[4*i], steps[4*i+1], steps[4 * i + 2], steps[4 * i + 3]);
				i++;
			}
			i = 1000;
			while (steps[4*i] != -1.f) {
				printf("child step [%d], t=%f\td=%f, cone: %f, pix: %f\n", i, steps[4 * i], steps[4 * i + 1], steps[4 * i + 2], steps[4 * i + 3]);
				i++;
			}

			printf("parent vs child final t: %f vs %f\n", debugData.parent_t, debugData.child_t);
		}

		if (false) {
			std::vector<float> jumpData(jumpBufferMaxElements, 0.f);
			std::vector<float> radiusData(jumpBufferMaxElements, 0.f);
			glGetNamedBufferSubData(jumpbuffer, 0, jumpBufferSize, jumpData.data());
			glGetNamedBufferSubData(radiusbuffer, 0, jumpBufferSize, radiusData.data());
			for (int i = 0; i < jumpBufferMaxElements; i++) {
				int b = tobin(i);
				int start = binto(b);
				int z = i - start;
				uvec2 coord = z2xy(uint(z));
				int dim = 1 << b;
				int size = dim * dim;

				int parent_size = size >> 2; 
				int parent = int(start - parent_size) + (z / 4);

				
					int pb = tobin(parent);
					int pstart = binto(pb);
					int pz = parent - pstart;
					uvec2 pcoord = z2xy(uint(pz));
					int pdim = 1 << pb;
					int psize = pdim * pdim;

				// printf("coord: (%u, %u) of %dx%d, pcoord: (%u, %u)\n", coord.x, coord.y, dim, dim, pcoord.x, pcoord.y);
				assert(coord.x / 2 == pcoord.x);
				assert(coord.y / 2 == pcoord.y);

				float myz = jumpData[i];
				float parentz = jumpData[parent];
				float myRadius = radiusData[i];
				float parentRadius = radiusData[parent];
				if (i > 0 && myRadius != .5f*parentRadius) {
					puts("fail");
				}
				if (myz < parentz) {
					printf("z[%d] %f < z[%d] %f!\n", i, myz, parent, parentz);
					puts("fail");
				}
			}
		}

		// DEBUG HACK: clear points every frame
		glClearNamedBufferData(pointBuffer, GL_R32F, GL_RED, GL_FLOAT, &zeroFloat);

		TimeStamp resolveTime;

		#define USE_BILINEAR_HISTORY_SAMPLE 0

		if (!draw)
			// the graphics program version of createProgram() takes 5 sources; vertex, control, evaluation, geometry, fragment
			draw = createProgram(
				GLSL(460,
					void main() {
						// note that nobody forces you to build VAOs and VBOs; you can write
						// arbitrary vertex shaders as long as they output gl_Position
						gl_Position = vec4(gl_VertexID == 1 ? 4. : -1., gl_VertexID == 2 ? 4. : -1., -.5, 1.);
					}
				),
				"", "", "",
					GLSL(460,
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

					layout(std140) uniform cameraArray { CameraParams cameras[2]; };
					layout(std430) buffer colorBuffer { uint colors[]; };
					layout(std430) buffer sampleWeightBuffer { uint sampleWeights[]; };
					layout(r8) uniform image2D edgebuffer;
					layout(std430) buffer jumpBuffer { float jumps[]; };
					layout(std430) buffer radiusBuffer { float radiuses[]; };
                    layout(std430) buffer uvBuffer { vec2 debug_uvs[]; };

					out vec4 outColor;
					uniform ivec2 screenSize;
					uniform int frame;
					uniform ivec3 noiseOffset;
					uniform sampler2DArray noiseTextures;

					// https://gamedev.stackexchange.com/a/148088
					vec3 linearToSRGB(vec3 linearRGB)
					{
						bvec3 cutoff = lessThan(linearRGB, vec3(0.0031308));
						vec3 higher = vec3(1.055)*pow(linearRGB, vec3(1.0 / 2.4)) - vec3(0.055);
						vec3 lower = linearRGB * vec3(12.92);

						return mix(higher, lower, cutoff);
					}

					void getCameraProjection(CameraParams cam, vec2 uv, out vec3 outPos, out vec3 outDir) {
						outPos = cam.pos + cam.dir + (uv.x - 0.5) * cam.right + (uv.y - 0.5) * cam.up;
						outDir = normalize(outPos - cam.pos);
					}

					vec3 getNoise(ivec2 coord)
					{
						return texelFetch(noiseTextures,
							ivec3((coord.x + noiseOffset.x) % 64, (coord.y + noiseOffset.y) % 64, noiseOffset.z), 0).rgb;
					}

					// Maps a ray index "i" into a bin index.
					int tobin(int i)
					{
						return findMSB(3 * i + 1) >> 1;
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
						return uvec2(z2x_1(z), z2x_1(z >> 1));
					}

					/**
					* Interleave lower 16 bits of x and y, so the bits of x
					* are in the even positions and bits from y in the odd;
					* z gets the resulting 32-bit Morton Number.
					* x and y must initially be less than 65536.
					*
					* Source: http://graphics.stanford.edu/~seander/bithacks.html
					*/
					uint xy2z(uint x, uint y) {
						uint B[] = { 0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF };
						uint S[] = { 1, 2, 4, 8 };

						x = (x | (x << S[3])) & B[3];
						x = (x | (x << S[2])) & B[2];
						x = (x | (x << S[1])) & B[1];
						x = (x | (x << S[0])) & B[0];

						y = (y | (y << S[3])) & B[3];
						y = (y | (y << S[2])) & B[2];
						y = (y | (y << S[1])) & B[1];
						y = (y | (y << S[0])) & B[0];

						//uint z = x | (y << 1);
						return x | (y << 1);
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

                        return uv;
                    }

					void main() {
						int pixelIdx = screenSize.x * int(gl_FragCoord.y) + int(gl_FragCoord.x);
                        vec2 pixelUV = (gl_FragCoord.xy - vec2(.5)) / screenSize.xy;

						uvec3 icolor = uvec3(
							colors[3 * pixelIdx + 0],
							colors[3 * pixelIdx + 1],
							colors[3 * pixelIdx + 2]
						);

						float edgeFactor = imageLoad(edgebuffer, ivec2(gl_FragCoord.xy)).x;
						//edgeFactor = 0.; // DEBUG HACK: edgebuffer disabled

						uint weightAlphaPacked = sampleWeights[pixelIdx];
						uint fixedWeight = weightAlphaPacked >> 16;
						uint fixedAlpha = weightAlphaPacked & 0xffff;
						float weight = float(fixedWeight) / 1000.;
						float alpha = float(fixedAlpha) / 255.;

						alpha = pow(min(1., alpha / 2.), 0.1);
						//alpha = pow(min(1., alpha/1.), 1.0);
						if (edgeFactor == 0.) alpha = 1.;
						vec3 color = vec3(icolor) / 10000.0;
						if (weight > 0.) {
							color /= weight;
						}
						
						vec3 skyColor = vec3(0., 0.5, 1.);
						vec3 c = mix(skyColor, color, alpha);

						if (false) {
							int bin = 9;
							int start = binto(bin);

							int dim = 1 << bin;

							vec2 fpos = vec2(
								((gl_FragCoord.x) / (screenSize.x / dim)),
								((gl_FragCoord.y) / (screenSize.y / dim))
							);
							uvec2 pos = uvec2(fpos);
							vec2 border = fract(fpos);

							uint z = xy2z(pos.x, pos.y);
							uint ind = start + z;


							int parent_start = binto(bin - 1);
							int parent_dim = 1 << (bin - 1);

							vec2 parent_fpos = vec2(
								(gl_FragCoord.x) / (screenSize.x / parent_dim),
								(gl_FragCoord.y) / (screenSize.y / parent_dim)
							);

							vec2 parent_border = fract(parent_fpos);

							uvec2 parent_pos = uvec2(parent_fpos);
							uint parent_z = xy2z(parent_pos.x, parent_pos.y);
							//uint parent_ind = parent_start + parent_z;
							uint parent_ind = parent_start + z/4;

							//
							if (ind == 112) {
							//if (parent_ind == 27) {
								outColor = vec4(1., 0., 1., 1.);
								return;
							}

							float d = jumps[ind];
							float dp= jumps[parent_ind];
							float diff = d - dp; // should be positive

							if (diff >= 0.) {
								//c.gb *= vec2(pow(d / 5., 5.));
								c.g *= pow(d / 5., 5.) * pow(border.x * border.y, .2);
							} else {
                                c.r = pow(border.x * border.y, .5);
							}

                            if (false && (frame/30 % 2 == 0)) {
                                float worldRadius = radiuses[parent_ind];
                                float uvRadius = worldRadius / (length(cameras[1].right));
                                vec2 uv = debug_uvs[parent_ind];
                                float dist = length(pixelUV - uv);
                                if (dist < uvRadius) {
                                    c.rg = vec2(pow(0.005/dist, 1.0), 0.);
                                }
                                if (dist < uvRadius / sqrt(2.)) {
                                    c.rg = c.gr;
                                }
                            }

                            if (d == 0.)
                            {
                                //c.rgb = vec3(1., 0., 0.);
                            }

                            //c.rgb = vec3(pow(d / 5., 5.));
                            //c.rgb = vec3(parent_fpos/80., 0.);
						}

						outColor = vec4(linearToSRGB(c.rgb), 1.);
						//outColor= vec4(vec3(alpha == 0.), 1.);

						vec3 noise = getNoise(ivec2(gl_FragCoord.xy));
						//outColor = vec4(noise, 1.);

						// Clear the accumulation buffer
						colors[3 * pixelIdx + 0] = 0;
						colors[3 * pixelIdx + 1] = 0;
						colors[3 * pixelIdx + 2] = 0;
						sampleWeights[pixelIdx] = 0;
						// Clear the edge buffer
						imageStore(edgebuffer, ivec2(gl_FragCoord.xy), vec4(0.));
					}
				)
			);

		glUseProgram(draw);

		glUniform1i("frame", frame);
		glUniform3i("noiseOffset", rand() % 64, rand() % 64, noiseLayer);
		glUniform1f("secs", secs);
		glUniform2i("screenSize", screenw, screenh);
		bindTexture("noiseTextures", noiseTextures);
		bindBuffer("colorBuffer", colorBuffer);
		bindImage("edgebuffer", 0, edgebuffer, GL_READ_WRITE, GL_R8); // TODO should be just GL_WRITE
		bindBuffer("sampleWeightBuffer", sampleWeightBuffer);
		bindBuffer("jumpBuffer", jumpbuffer);
		bindBuffer("radiusBuffer", radiusbuffer);
		bindBuffer("uvBuffer", uvbuffer);
		bindBuffer("cameraArray", cameraData);
		glDrawArrays(GL_TRIANGLES, 0, 3);

		// here we're just using two timestamps, but you could of course measure intermediate timings as well
		TimeStamp end;

		// print the timing (word of warning; this forces a cpu-gpu synchronization)
		font.drawText(L"Total: " + std::to_wstring(end - start), 10.f, 10.f, 15.f); // text, x, y, font size
		font.drawText(L"Draw: " + std::to_wstring(drawTime - start), 10.f, 25.f, 15.f);
		font.drawText(L"Splat: " + std::to_wstring(splatTime - drawTime), 10.f, 40.f, 15.f);
		font.drawText(L"PostProc: " + std::to_wstring(end - splatTime), 10.f, 55.f, 15.f);
		font.drawText(L"Points: " + std::to_wstring(pointsSplatted / 1000. / 1000.) + L" M", 200.f, 10.f, 15.f);

		// this actually displays the rendered image
		swapBuffers();

		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

		std::swap(cameras[0], cameras[1]);
		frame++;
	}
	return 0;
}
