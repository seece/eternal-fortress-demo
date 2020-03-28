
#include "testbench.h"
#include <cinttypes>

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

const int screenw = 1024, screenh = 1024;
static constexpr int SAMPLES_PER_PIXEL = 1;
static constexpr GLuint SAMPLE_BUFFER_TYPE = GL_RGBA16F;
static constexpr GLuint JITTER_BUFFER_TYPE = GL_RG8;

static void cameraPath(float t, CameraParameters& cam)
{
	float tt = t * 0.2f;
	//cam.pos = vec3(0.5f*sin(tt), 0.f, 6.f + 0.5f*cos(tt));
	cam.pos = vec3(1., 0., 4.f + 2.5f*cos(tt));
	cam.dir = normalize(vec3(0.5f*cos(tt*0.5f), 0.2f*sin(tt), -1.f));
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

constexpr int MAX_POINT_COUNT = 10 * 1000 * 1000;

int main() {

	// create window and context. can also do fullscreen and non-visible window for compute only
	OpenGL context(screenw, screenh, "raymarcher");
	
	// load a font to draw text with -- any system font or local .ttf file should work
	Font font(L"Consolas");

	// shader variables; could also initialize them here, but it's often a good idea to
	// do that at the callsite (so input/output declarations are close to the bind code)
	Program march, draw, sampleResolve, headerUpdate, pointSplat;

	Texture<GL_TEXTURE_2D_ARRAY> abuffer;
	Texture<GL_TEXTURE_2D> gbuffer;
	Texture<GL_TEXTURE_2D> zbuffer;
	Texture<GL_TEXTURE_2D_ARRAY> samplebuffer;
	Texture<GL_TEXTURE_2D_ARRAY> jitterbuffer;
	Buffer cameraData;
	Buffer pointBufferHeader;
	Buffer pointBuffer;
	Buffer colorBuffer, sampleWeightBuffer;

	setWrapToClamp(abuffer);
	setWrapToClamp(gbuffer);
	setWrapToClamp(zbuffer);
	setWrapToClamp(samplebuffer);

	int renderw = screenw, renderh = screenh;

	glTextureStorage3D(abuffer, 1, GL_RGBA32F, screenw, screenh, 2);
	glTextureStorage2D(gbuffer, 1, GL_RGBA32F, renderw, renderh);
	glTextureStorage2D(zbuffer, 1, GL_R32F, renderw, renderh);
	glTextureStorage3D(samplebuffer, 1, SAMPLE_BUFFER_TYPE, renderw, renderh, SAMPLES_PER_PIXEL);
	glTextureStorage3D(jitterbuffer, 1, JITTER_BUFFER_TYPE, renderw, renderh, SAMPLES_PER_PIXEL);

	int abuffer_read_layer = 0, frame = 0;
	CameraParameters cameras[2] = {};
	glNamedBufferStorage(cameraData, sizeof(cameras), NULL, GL_DYNAMIC_STORAGE_BIT);
	glNamedBufferStorage(pointBufferHeader, 4, NULL, GL_DYNAMIC_STORAGE_BIT); // TODO read bit only for debugging
	glNamedBufferStorage(pointBuffer, sizeof(RgbPoint) * MAX_POINT_COUNT, NULL, GL_DYNAMIC_STORAGE_BIT); // TODO read bit only for debugging
	glNamedBufferStorage(colorBuffer, screenw * screenh * 3 * sizeof(int), NULL, 0);
	glNamedBufferStorage(sampleWeightBuffer, screenw * screenh * sizeof(int), NULL, 0);

	int zero = 0;
	glClearNamedBufferData(pointBufferHeader, GL_R32I, GL_RED_INTEGER, GL_INT, &zero);
	glClearNamedBufferData(pointBuffer, GL_R32I, GL_RED_INTEGER, GL_INT, &zero);
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

	while (loop()) // loop() stops if esc pressed or window closed
	{
		// timestamp objects make gl queries at those locations; you can substract them to get the time
		TimeStamp start;
		float secs = fmod(frame / 60.f, 2.0) + 21.;
		float futureInterval = 1. / 60.f;
		cameraPath(secs + futureInterval, cameras[1]);
		glNamedBufferSubData(cameraData, 0, sizeof(cameras), &cameras);

		glDisable(GL_BLEND);

		if (!march)
			march = createProgram("shaders/marcher.glsl");

		glUseProgram(march);

		glUniform1i("frame", frame);
		glUniform1f("secs", secs);
		glUniform1i("pointBufferMaxElements", pointBufferMaxElements);
		bindBuffer("cameraArray", cameraData);
		bindBuffer("pointBufferHeader", pointBufferHeader);
		bindBuffer("pointBuffer", pointBuffer);
		//bindImage("gbuffer", 0, gbuffer, GL_WRITE_ONLY, GL_RGBA32F);
		bindImage("zbuffer", 0, zbuffer, GL_WRITE_ONLY, GL_R32F);
		bindImage("samplebuffer", 0, samplebuffer, GL_WRITE_ONLY, SAMPLE_BUFFER_TYPE);
		bindImage("jitterbuffer", 0, jitterbuffer, GL_WRITE_ONLY, JITTER_BUFFER_TYPE);
		// the arguments of dispatch are the numbers of thread blocks in each direction;
		// since our local size is 16x16x1, we'll get 1024x1024x1 threads total, just enough
		// for our image
		glDispatchCompute(64, 64, 1);

		// we're writing to an image in a shader, so we should have a barrier to ensure the writes finish
		// before the next shader call (wasn't an issue on my hardware in this case, but you should always make sure
		// to place the correct barriers when writing from compute shaders and reading in subsequent shaders)
		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT | GL_ATOMIC_COUNTER_BARRIER_BIT);

		int currentWriteOffset = -1;
		glGetNamedBufferSubData(pointBufferHeader, 0, 4, &currentWriteOffset);
		printf("currentWriteOffset: %d\n", currentWriteOffset);

		TimeStamp drawTime;

		if (!headerUpdate) {
			headerUpdate = createProgram(
				GLSL(460,
					layout(local_size_x = 16, local_size_y = 16) in;

					layout(std140) buffer pointBufferHeader {
						int currentWriteOffset;
					};

					uniform int pointBufferMaxElements;

					void main() {
						currentWriteOffset = currentWriteOffset % pointBufferMaxElements;
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

			uniform int pointBufferMaxElements;
			uniform int numberOfPointsToSplat;
			uniform ivec2 screenSize;

			layout(std140) uniform cameraArray {
				CameraParams cameras[2];
			};

			layout(std430) buffer pointBufferHeader {
				int currentWriteOffset;
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

				if (any(lessThan(camSpace, vec3(0.))))
				{
					return;
				}

				if (any(greaterThan(camSpace.xy, vec2(1.)))) {
					return;
				}

				int x = int(camSpace.x * screenSize.x);
				int y = int(camSpace.y * screenSize.y);

				int pixelIdx = screenSize.x * y + x;

				float distance = length(pos.xyz - cameras[1].pos);
				float fog = pow(min(1., distance / 10.), 4.0);
				c = mix(c, vec3(0.5, 0., 0.), fog);

				c = c / (vec3(1.) + c);
				c = clamp(c, vec3(0.), vec3(10.));

				float weight = max(1.0, min(3e3,
					1. / (pow(camSpace.z / 2., 2.) + 0.001)));

				uvec3 icolor = uvec3(weight * 1000 * c);
				atomicAdd(colors[3 * pixelIdx + 0], icolor.r);
				atomicAdd(colors[3 * pixelIdx + 1], icolor.g);
				atomicAdd(colors[3 * pixelIdx + 2], icolor.b);
				atomicAdd(sampleWeights[pixelIdx], int(1000 * weight));
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
		bindBuffer("colorBuffer", colorBuffer);
		bindBuffer("sampleWeightBuffer", sampleWeightBuffer);
		bindBuffer("cameraArray", cameraData);
		glDispatchCompute(numberOfPointsToSplat / 128 / 1, 1, 1);

		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT | GL_ATOMIC_COUNTER_BARRIER_BIT);

		TimeStamp splatTime;

		if (false) {
			if (!sampleResolve) {
				sampleResolve = createProgram(
					GLSL(460,
						layout(local_size_x = 16, local_size_y = 16) in;

				uniform sampler2DArray samplebuffer;
				uniform sampler2DArray jitterbuffer;
				uniform sampler2D zbuffer;
				layout(rgba32f) uniform image2D gbuffer;

				void main() {
					ivec2 ij = ivec2(gl_GlobalInvocationID.xy);
					int samplesPerPixel = textureSize(samplebuffer, 0).z;
					vec3 accum = vec3(0.);
					float totalWeight = 1e-6;

					float centerz = texelFetch(zbuffer, ij, 0).x;

#ifdef USE_PROPER_FILTER

					float aroundzMax = 0.;
					float aroundzMin = 1e9;
					vec3 around = vec3(0.);
					for (int i = 0; i < samplesPerPixel; i++) {
						for (int y = -1; y <= 1; y++) {
							for (int x = -1; x <= 1; x++) {
								ivec2 delta = ivec2(x, y);
								ivec2 coord = ij + delta;

								vec4 c = texelFetch(samplebuffer, ivec3(coord, i), 0);
								float z = texelFetch(zbuffer, coord, 0).x;

								// Tonemap colors already here
								c.rgb = c.rgb / (vec3(1.) + c.rgb);

								vec2 jitter = texelFetch(jitterbuffer, ivec3(coord, i), 0).xy;
								jitter -= vec2(0.5);

								if (x == 0 && y == 0) {
									centerz = z;
								}
								else {
									aroundzMax = max(aroundzMax, z);
									aroundzMin = min(aroundzMin, z);
									around += c.rgb;
								}

								vec2 sampleCoord = vec2(x, y) + jitter;
								float d = length(sampleCoord);
								float weight = max(0., 1.25 - length(d));
								//weight = 1.0;
								//weight = exp(-2.29 * pow(length(sampleCoord), 2.)); // PRMan Gaussian fit to Blackman-Harris 3.
								accum += weight * c.rgb;
								totalWeight += weight;
							}
						}
					}
#else 
					for (int i = 0; i < samplesPerPixel; i++) {
						vec4 c = texelFetch(samplebuffer, ivec3(ij, i), 0);
						float z = texelFetch(zbuffer, ij, 0).x;
						accum += c.rgb;
						centerz = min(centerz, z);
					}
					totalWeight = samplesPerPixel;
#endif

					accum /= totalWeight;

					float outz = centerz;

					/*if (centerz > aroundzMax) {
						accum = around / 8.;
						outz = aroundzMin;
					}*/

					imageStore(gbuffer, ij, vec4(accum, outz));
				}
				)
				);
			}

			glUseProgram(sampleResolve);
			bindImage("gbuffer", 0, gbuffer, GL_WRITE_ONLY, GL_RGBA32F);
			bindTexture("zbuffer", zbuffer);
			bindTexture("samplebuffer", samplebuffer);
			bindTexture("jitterbuffer", jitterbuffer);
			glDispatchCompute(64, 64, 1);

			glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
		}

		TimeStamp resolveTime;

		// flip the ping-pong flag
		abuffer_read_layer = 1 - abuffer_read_layer;

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

					layout(std140) uniform cameraArray { CameraParams cameras[2]; };
					layout(std430) buffer colorBuffer { uint colors[]; };
					layout(std430) buffer sampleWeightBuffer { uint sampleWeights[]; };

					out vec4 outColor;
					uniform ivec2 screenSize;
					uniform int frame;

					void main() {
						int pixelIdx = screenSize.x * int(gl_FragCoord.y) + int(gl_FragCoord.x);

						uvec3 icolor = uvec3(
							colors[3 * pixelIdx + 0],
							colors[3 * pixelIdx + 1],
							colors[3 * pixelIdx + 2]
						);

						float totalWeight = float(sampleWeights[pixelIdx]) / 1000.;
						vec3 color = vec3(icolor) / 1000.;
						color /= totalWeight;

						vec3 c = color;
						outColor = vec4(linearToSRGB(c.rgb), 1.);

						// Clear the accumulation buffer
						colors[3 * pixelIdx + 0] = 0;
						colors[3 * pixelIdx + 1] = 0;
						colors[3 * pixelIdx + 2] = 0;
						sampleWeights[pixelIdx] = 0;
					}
				)
			);

		glUseProgram(draw);

		glUniform1i("frame", frame);
		glUniform1f("secs", secs);
		glUniform2i("screenSize", screenw, screenh);
		bindBuffer("colorBuffer", colorBuffer);
		bindBuffer("sampleWeightBuffer", sampleWeightBuffer);
		bindBuffer("cameraArray", cameraData);
		glDrawArrays(GL_TRIANGLES, 0, 3);

		// here we're just using two timestamps, but you could of course measure intermediate timings as well
		TimeStamp end;

		// print the timing (word of warning; this forces a cpu-gpu synchronization)
		font.drawText(L"Total: " + std::to_wstring(end - start), 10.f, 10.f, 15.f); // text, x, y, font size
		font.drawText(L"Draw: " + std::to_wstring(drawTime - start), 10.f, 25.f, 15.f);
		font.drawText(L"Splat: " + std::to_wstring(splatTime - drawTime), 10.f, 40.f, 15.f);
		font.drawText(L"PostProc: " + std::to_wstring(end - splatTime), 10.f, 55.f, 15.f);

		// this actually displays the rendered image
		swapBuffers();

		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

		std::swap(cameras[0], cameras[1]);
		frame++;
	}
	return 0;
}
