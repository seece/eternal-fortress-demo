
#include "testbench.h"


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


int main() {

	// create window and context. can also do fullscreen and non-visible window for compute only
	OpenGL context(screenw, screenh, "raymarcher");
	
	// load a font to draw text with -- any system font or local .ttf file should work
	Font font(L"Consolas");

	// shader variables; could also initialize them here, but it's often a good idea to
	// do that at the callsite (so input/output declarations are close to the bind code)
	Program march, draw, sampleResolve;

	Texture<GL_TEXTURE_2D_ARRAY> abuffer;
	Texture<GL_TEXTURE_2D> gbuffer;
	Texture<GL_TEXTURE_2D> zbuffer;
	Texture<GL_TEXTURE_2D_ARRAY> samplebuffer;
	Texture<GL_TEXTURE_2D_ARRAY> jitterbuffer;
	Buffer cameraData;

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


	while (loop()) // loop() stops if esc pressed or window closed
	{
		// timestamp objects make gl queries at those locations; you can substract them to get the time
		TimeStamp start;
		float secs = frame / 60.f;
		cameraPath(fmod(secs, 2.)+30., cameras[1]);
		glNamedBufferSubData(cameraData, 0, sizeof(cameras), &cameras);

		glDisable(GL_BLEND);

		if (!march)
			march = createProgram("shaders/marcher.glsl");

		glUseProgram(march);

		glUniform1i("frame", frame);
		glUniform1f("secs", secs);
		bindBuffer("cameraArray", cameraData);
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
		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

		TimeStamp drawTime;

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

		TimeStamp resolveTime;

		// flip the ping-pong flag
		abuffer_read_layer = 1 - abuffer_read_layer;

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

					layout(std140) uniform cameraArray {
						CameraParams cameras[2];
					};

					// https://gamedev.stackexchange.com/a/148088
					vec3 linearToSRGB(vec3 linearRGB)
					{
						bvec3 cutoff = lessThan(linearRGB, vec3(0.0031308));
						vec3 higher = vec3(1.055)*pow(linearRGB, vec3(1.0 / 2.4)) - vec3(0.055);
						vec3 lower = linearRGB * vec3(12.92);

						return mix(higher, lower, cutoff);
					}

					// source: https://github.com/playdeadgames/temporal/blob/4795aa0007d464371abe60b7b28a1cf893a4e349/Assets/Shaders/TemporalReprojection.shader#L122
					vec4 clip_aabb(vec3 aabb_min, vec3 aabb_max, vec4 p, vec4 q)
					{
						// note: only clips towards aabb center (but fast!)
						vec3 p_clip = vec3(0.5) * (aabb_max + aabb_min);
						vec3 e_clip = vec3(0.5) * (aabb_max - aabb_min) + vec3(0.00000001f);

						vec4 v_clip = q - vec4(p_clip, p.w);
						vec3 v_unit = v_clip.xyz / e_clip;
						vec3 a_unit = abs(v_unit);
						float ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));

						if (ma_unit > 1.0)
							return vec4(p_clip, p.w) + v_clip / ma_unit;
						else
							return q;// point inside aabb
					}

					void getCameraProjection(CameraParams cam, vec2 uv, out vec3 outPos, out vec3 outDir) {
						outPos = cam.pos + cam.dir + (uv.x - 0.5) * cam.right + (uv.y - 0.5) * cam.up;
						outDir = normalize(outPos - cam.pos);
					}

					vec2 reprojectPoint(CameraParams cam, vec3 p) {
						vec3 op = p - cam.pos;
						float n = length(cam.dir);
						float z = dot(cam.dir, op) / n;
						vec3 pp = (op * n) / z;
						vec2 plane = vec2(
							dot(pp, cam.right) / dot(cam.right, cam.right),
							dot(pp, cam.up) / dot(cam.up, cam.up)
						);
						return plane + vec2(0.5);
					}

					float rgb2Luminance(vec3 rgb) {
						return dot(rgb, vec3(0.2126, 0.7152, 0.0722));
					}

					uniform sampler2D gbuffer;
					uniform sampler2D zbuffer;
					uniform sampler2DArray abuffer;
					layout(rgba32f) uniform image2DArray abuffer_image;
					out vec4 col;
					uniform int abuffer_read_layer;
					uniform int frame;

					vec4 sampleHistoryBuffer(vec2 uv) {
						// return texture(abuffer, vec3(uv, abuffer_read_layer));
						vec2 uvScreen = uv * textureSize(abuffer, 0).xy;
						//uvScreen -= vec2(0.5);

						/*
						vec2 uv2 = (uvScreen - vec2(.5));
						ivec2 uv2pix = ivec2(uv2);
						vec2 w = fract(uv2);
						vec4 col = vec4(0.);
						col += (1. - w.x) * (1. - w.y) * texelFetch(abuffer, ivec3(uv2pix, abuffer_read_layer), 0);
						col += w.x * (1. - w.y) * texelFetch(abuffer, ivec3(uv2pix + vec2(1., 0), abuffer_read_layer), 0);
						col += (1. - w.x) * w.y * texelFetch(abuffer, ivec3(uv2pix + vec2(0., 1.), abuffer_read_layer), 0);
						col += w.x * w.y * texelFetch(abuffer, ivec3(uv2pix + vec2(1., 1.), abuffer_read_layer), 0);
						return col;
						*/
						
						//return texture(abuffer, vec3(uv, abuffer_read_layer), 0);

						ivec2 pixelCoords = ivec2(uvScreen - vec2(0.5));
						vec2 subPixel = fract(uvScreen - vec2(0.5));

						float totalWeight = 0.;
						vec3 color = vec3(0.);
						float z = 1e9;

						for (int y = -1; y <= 1; y++) {
							for (int x = -1; x <= 1; x++) {

								ivec2 delta = ivec2(x, y);

								// "pos" in the same coordinate system as "subPixel."
								// This means e.g. (-1, 0) is at the center of the texel on the left.
								vec2 pos = vec2(x, y);

								vec4 c = texelFetch(abuffer, ivec3(pixelCoords + delta, abuffer_read_layer), 0);
								
								// Tent filter based on the distance from the pixel center.
								float d = length(pos - subPixel);
								float weight = max(0., 1.0 - d); //TODO 1.0-d is like bilinear tap
								//weight = pow(weight, .5);
								//weight = 1.;

								color += weight * c.rgb;
								totalWeight += abs(weight);
								z += c.w;
								//z = min(z, c.w);
							}
						}

						color /= totalWeight;
						z /= totalWeight;
						
						return vec4(color, z);
					}

					void main() {
						//vec2 pixelCoord = ivec2(gl_FragCoord.xy + vec2(0.5));
						vec4 c1 = texelFetch(gbuffer, ivec2(gl_FragCoord.xy), 0);
						//float z1 = texelFetch(zbuffer, ivec2(gl_FragCoord.xy), 0).x;
						float z1 = c1.w;
						const ivec2[8] deltas = {
							ivec2(0, -1),  // direct neighbors
							ivec2(-1, 0),
							ivec2(1,  0),
							ivec2(0,  1),
							ivec2(-1, -1), // diagonals
							ivec2(1, -1),
							ivec2(-1, 1),
							ivec2(1,  1),
						};

						float minz = 1e9;

						for (int i = 0; i < 8; i++) {
							float z = texelFetch(zbuffer, ivec2(gl_FragCoord.xy) + deltas[i], 0).x;
							minz = min(minz, z);
						}
						//z1 = min(z1, minz); // TODO is this necessary

						vec3 minbox = c1.rgb;
						vec3 maxbox = c1.rgb;
						vec3 cavg = c1.rgb;
						for (int i = 0; i < 8; i++) {
							vec4 cn = texelFetch(gbuffer, ivec2(gl_FragCoord.xy) + deltas[i], 0);
							minbox = min(minbox, cn.rgb);
							maxbox = max(maxbox, cn.rgb);
							cavg += cn.rgb;
						}
						cavg /= 9.;
						// compute also cross average

						ivec2 res = textureSize(gbuffer, 0).xy;

						vec3 rayStartPos1, rayDir1;
						vec2 uv1 = vec2(gl_FragCoord.xy) / res;
						getCameraProjection(cameras[1], uv1, rayStartPos1, rayDir1);
						vec3 world1 = rayStartPos1 + rayDir1 * z1;

						vec2 uv0 = reprojectPoint(cameras[0], world1);
						vec4 c0 = sampleHistoryBuffer(uv0);
						float z0 = c0.w;
						vec3 rayStartPos0, rayDir0;
						getCameraProjection(cameras[0], uv0, rayStartPos0, rayDir0);
						vec3 world0 = rayStartPos0 + rayDir0 * z0;

						float worldSpaceDist = length(world0 - world1);
						float screenSpaceDist = length((uv1 - uv0) * res);

						//c0.rgb = clamp(c0.rgb, minbox, maxbox);
						c0.rgb = clip_aabb(minbox, maxbox, vec4(clamp(cavg, minbox, maxbox), 0.), vec4(c0.rgb, 0.)).rgb;

						// feedback weight from unbiased luminance diff (t.lottes)
						// https://github.com/playdeadgames/temporal/blob/4795aa0007d464371abe60b7b28a1cf893a4e349/Assets/Shaders/TemporalReprojection.shader#L313

						float lum0 = rgb2Luminance(c0.rgb); // last frame's
						float lum1 = rgb2Luminance(c1.rgb); // this frame's
						float unbiased_diff = abs(lum0 - lum1) / max(lum0, max(lum1, 0.2));
						float unbiased_weight = 1.0 - unbiased_diff;
						float unbiased_weight_sqr = unbiased_weight * unbiased_weight;
						float feedback = mix(0.0, 1.0, unbiased_weight_sqr);
						feedback = clamp(0.95 - screenSpaceDist/20., 0., 1.);
						//feedback = 0.9;
						//feedback = 1. - 1./frame;
						//feedback = max(0., feedback - worldSpaceDist*100.);

						if (frame == 0) feedback = 0;
						//if (z1 >= 1e9) feedback = 0;
						//vec3 c = c1.rgb;
						//c.rgb = c1.rgb;

						// Tone map c1, c0 is already in LDR
						// c0.rgb = c0.rgb / (vec3(1.) + c0.rgb);
						// c1.rgb = c1.rgb / (vec3(1.) + c1.rgb);
						
						vec3 c = feedback * max(vec3(0.), c0.xyz) + (1 - feedback) * max(vec3(0.), c1.xyz);

						col.rgb = c.rgb;
						col = vec4(linearToSRGB(col.rgb), 1.);

						// Inverse tone map, store linear values
						// c = c / (vec3(1.) - c);

						imageStore(abuffer_image, ivec3(gl_FragCoord.xy, 1 - abuffer_read_layer), vec4(c, z1));
					}
				)
			);

		glUseProgram(draw);

		bindTexture("gbuffer", gbuffer);
		bindTexture("zbuffer", zbuffer);
		bindTexture("abuffer", abuffer);
		bindImage("abuffer_image", 0, abuffer, GL_WRITE_ONLY, GL_RGBA32F);
		glUniform1i("abuffer_read_layer", abuffer_read_layer);
		glUniform1i("frame", frame);
		glUniform1f("secs", secs);
		bindBuffer("cameraArray", cameraData);
		glDrawArrays(GL_TRIANGLES, 0, 3);

		// here we're just using two timestamps, but you could of course measure intermediate timings as well
		TimeStamp end;

		// print the timing (word of warning; this forces a cpu-gpu synchronization)
		font.drawText(L"Total: " + std::to_wstring(end - start), 10.f, 10.f, 15.f); // text, x, y, font size
		font.drawText(L"Draw: " + std::to_wstring(drawTime - start), 10.f, 25.f, 15.f);
		font.drawText(L"Resolve: " + std::to_wstring(resolveTime - drawTime), 10.f, 40.f, 15.f); 
		font.drawText(L"PostProc: " + std::to_wstring(end - resolveTime), 10.f, 55.f, 15.f);

		// this actually displays the rendered image
		swapBuffers();

		std::swap(cameras[0], cameras[1]);
		frame++;
	}
	return 0;
}
