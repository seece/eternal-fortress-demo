
#include "testbench.h"


struct CameraParameters {
	vec3 pos;
	float padding;
	vec3 dir;
	float zoom;
	vec3 up;
	float padding2;
	vec3 right;
	float padding3;
};

const int screenw = 1024, screenh = 1024;
static constexpr int SAMPLES_PER_PIXEL = 1;
static constexpr GLuint SAMPLE_BUFFER_TYPE = GL_RGBA16F;

static void cameraPath(float t, CameraParameters& cam)
{
	float tt = t * 0.2f;
	cam.pos = vec3(0.5f*sin(tt), 0.f, 6.f + 0.5f*cos(tt));
	cam.dir = normalize(vec3(0.5f*cos(tt*0.5f), 0.2f*sin(tt), -1.f));
	cam.zoom = 0.9f;
	cam.right = normalize(cross(cam.dir, vec3(0.f, 1.f, 0.f)));
	cam.up = cross(cam.dir, cam.right);
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

	int abuffer_read_layer = 0, frame = 0;
	CameraParameters cameras[2] = {};
	glNamedBufferStorage(cameraData, sizeof(cameras), NULL, GL_DYNAMIC_STORAGE_BIT);


	while (loop()) // loop() stops if esc pressed or window closed
	{
		// timestamp objects make gl queries at those locations; you can substract them to get the time
		TimeStamp start;
		float secs = frame / 60.f;
		cameraPath(secs, cameras[1]);
		glNamedBufferSubData(cameraData, 0, sizeof(cameras), &cameras);

		if (!march)
			march = createProgram("shaders/marcher.glsl");

		glUseProgram(march);

		glUniform1i("frame", frame);
		glUniform1f("secs", secs);
		bindBuffer("cameraArray", cameraData);
		//bindImage("gbuffer", 0, gbuffer, GL_WRITE_ONLY, GL_RGBA32F);
		bindImage("zbuffer", 0, zbuffer, GL_WRITE_ONLY, GL_R32F);
		bindImage("samplebuffer", 0, samplebuffer, GL_WRITE_ONLY, SAMPLE_BUFFER_TYPE);
		// the arguments of dispatch are the numbers of thread blocks in each direction;
		// since our local size is 16x16x1, we'll get 1024x1024x1 threads total, just enough
		// for our image
		glDispatchCompute(64, 64, 1);

		// we're writing to an image in a shader, so we should have a barrier to ensure the writes finish
		// before the next shader call (wasn't an issue on my hardware in this case, but you should always make sure
		// to place the correct barriers when writing from compute shaders and reading in subsequent shaders)
		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
		glMemoryBarrier(GL_ATOMIC_COUNTER_BARRIER_BIT); // TODO Is this needed for integer texture atomic max?

		TimeStamp drawTime;

		if (!sampleResolve) {
			sampleResolve = createProgram(
				GLSL(460,
					layout(local_size_x = 16, local_size_y = 16) in;

					vec2 unpackJitter(float d)
					{
						uint c = floatBitsToUint(d);
						uvec2 b = uvec2((c >> 8) & 0xff, c & 0xff);
						vec2 a = vec2(b) / 255;
						return a - vec2(0.5);
					}

					uniform sampler2DArray samplebuffer;
					uniform sampler2D zbuffer;
					layout(rgba32f) uniform image2D gbuffer;

					void main() {
						ivec2 ij = ivec2(gl_GlobalInvocationID.xy);
						int samplesPerPixel = textureSize(samplebuffer, 0).z;
						vec3 accum = vec3(0.);
						float totalWeight = 0.;
						//ivec2 dp = ivec2(jitter.x < 0 ? -1 : 1, jitter.y < 0 ? -1 : 1);

						for (int y = -1; y <= 1; y++) {
							for (int x = -1; x <= 1; x++) {
								ivec2 delta = ivec2(x, y);
								ivec2 coord = ij + delta;
								for (int i = 0; i < samplesPerPixel; i++) {
									vec4 c = texelFetch(samplebuffer, ivec3(coord, i), 0);
									vec2 jitter = unpackJitter(floatBitsToUint(c.w));
									
									vec2 sampleCoord = vec2(x, y) + jitter;
									float weight = max(0., 1. - length(sampleCoord));
									accum += weight * c.rgb;
									totalWeight += weight;
								}
							}
						}

						accum /= totalWeight;
						imageStore(gbuffer, ij, vec4(accum, 0.));
					}
			)
			);
		}

		glUseProgram(sampleResolve);
		bindImage("gbuffer", 0, gbuffer, GL_WRITE_ONLY, GL_RGBA32F);
		bindTexture("samplebuffer", samplebuffer);
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
						uv /= cam.zoom;
						// vec3 right = cross(cam.dir, vec3(0., 1., 0.));
						// vec3 up = cross(cam.dir, right);
						outPos = cam.pos + cam.dir + (uv.x - 0.5) * cam.right + (uv.y - 0.5) * cam.up;
						outDir = normalize(outPos - cam.pos);
					}

					vec2 reprojectPoint(CameraParams cam, vec3 p) {
						vec3 op = p - cam.pos;
						float z = dot(cam.dir, op);
						vec3 pp = op / z;
						vec2 plane = vec2(dot(cam.right, pp), dot(cam.up, pp));
						return cam.zoom * (plane + vec2(0.5));
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

					void main() {
						vec4 c1 = texelFetch(gbuffer, ivec2(gl_FragCoord.xy), 0);
						float z1 = texelFetch(zbuffer, ivec2(gl_FragCoord.xy), 0).x;
						const ivec2[8] deltas = {
							ivec2(-1, -1),
							ivec2(0, -1),
							ivec2(1, -1),
							ivec2(-1,  0),
							ivec2(1,  0),
							ivec2(-1,  1),
							ivec2(0,  1),
							ivec2(1,  1),
						};

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

						vec3 rayStartPos1, rayDir1;
						vec2 uv1 = vec2(gl_FragCoord.xy) / 1024.;
						getCameraProjection(cameras[1], uv1, rayStartPos1, rayDir1);
						vec3 world1 = cameras[1].pos + rayDir1 * z1;

						vec2 uv0 = reprojectPoint(cameras[0], world1);
						vec4 c0 = texture(abuffer, vec3(uv0, abuffer_read_layer));
						float z0 = c0.w; // NOTE: Linearly interpolated depth.
						vec3 rayStartPos0, rayDir0;
						getCameraProjection(cameras[0], uv0, rayStartPos0, rayDir0);
						vec3 world0 = cameras[0].pos + rayDir0 * z0;

						float worldSpaceDist = length(world0 - world1);

						//c0.rgb = clamp(c0.rgb, minbox, maxbox);
						//c0.rgb = clip_aabb(minbox, maxbox, vec4(clamp(cavg, minbox, maxbox), 0.), vec4(c0.rgb, 0.)).rgb;

						// feedback weight from unbiased luminance diff (t.lottes)
						// https://github.com/playdeadgames/temporal/blob/4795aa0007d464371abe60b7b28a1cf893a4e349/Assets/Shaders/TemporalReprojection.shader#L313

						float lum0 = rgb2Luminance(c0.rgb); // last frame's
						float lum1 = rgb2Luminance(c1.rgb); // this frame's
						float unbiased_diff = abs(lum0 - lum1) / max(lum0, max(lum1, 0.2));
						float unbiased_weight = 1.0 - unbiased_diff;
						float unbiased_weight_sqr = unbiased_weight * unbiased_weight;
						float feedback = mix(0.0, 1.0, unbiased_weight_sqr);
						feedback = 0.0;
						//feedback = max(0., feedback - worldSpaceDist*100.);

						if (frame == 0) feedback = 0;
						vec3 c = feedback * c0.xyz + (1 - feedback) * c1.xyz;
						imageStore(abuffer_image, ivec3(gl_FragCoord.xy, 1 - abuffer_read_layer), vec4(c, z1));

						c = c / (vec3(1.) + c);
						col = vec4(pow(c, vec3(1. / 2.2)), 1.);
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
