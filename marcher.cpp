
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

void cameraPath(float t, CameraParameters& cam)
{
	float tt = t * 0.2f;
	cam.pos = vec3(0.5f*sin(tt), 0.f, 6.f + 0.5f*cos(tt));
	cam.dir = normalize(vec3(0.5f*cos(tt*0.5f), 0.2f*sin(tt), -1.f));
	cam.zoom = 0.9f;
	cam.right = normalize(cross(cam.dir, vec3(0.f, 1.f, 0.f)));
	cam.up = cross(cam.dir, cam.right);
}


int main() {

	// create window and context. can also do fullscreen and non-visible window for compute only
	OpenGL context(screenw, screenh, "raymarcher");
	
	// load a font to draw text with -- any system font or local .ttf file should work
	Font font(L"Consolas");

	// shader variables; could also initialize them here, but it's often a good idea to
	// do that at the callsite (so input/output declarations are close to the bind code)
	Program march, draw;

	Texture<GL_TEXTURE_2D_ARRAY> abuffer;
	Texture<GL_TEXTURE_2D> gbuffer;
	Buffer cameraData;

	int renderw = screenw, renderh = screenh;

	glTextureStorage3D(abuffer, 1, GL_RGBA32F, screenw, screenh, 2);
	glTextureStorage2D(gbuffer, 1, GL_RGBA32F, renderw, renderh);

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
		// .. this includes images
		bindImage("gbuffer", 0, gbuffer, GL_WRITE_ONLY, GL_RGBA32F);
		// the arguments of dispatch are the numbers of thread blocks in each direction;
		// since our local size is 16x16x1, we'll get 1024x1024x1 threads total, just enough
		// for our image
		glDispatchCompute(64, 64, 1);

		// we're writing to an image in a shader, so we should have a barrier to ensure the writes finish
		// before the next shader call (wasn't an issue on my hardware in this case, but you should always make sure
		// to place the correct barriers when writing from compute shaders and reading in subsequent shaders)
		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

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

					uniform sampler2D gbuffer;
					uniform sampler2DArray abuffer;
					layout(rgba32f) uniform image2DArray abuffer_image;
					out vec4 col;
					uniform int abuffer_read_layer;
					uniform int frame;

					void main() {
						vec4 c1 = texelFetch(gbuffer, ivec2(gl_FragCoord.xy), 0);

						vec3 rayStartPos1, rayDir1;
						vec2 uv1 = vec2(gl_FragCoord.xy) / 1024.;
						getCameraProjection(cameras[1], uv1, rayStartPos1, rayDir1);
						vec3 world1 = cameras[1].pos + rayDir1 * c1.w;
						vec2 uv0 = reprojectPoint(cameras[0], world1);

						//vec4 c0 = texelFetch(abuffer, ivec3(vec2(0.5) + uv0 * textureSize(abuffer, 0).xy, abuffer_read_layer), 0);
						vec4 c0 = texture(abuffer, vec3(uv0, 0.));
						vec3 rayStartPos0, rayDir0;
						getCameraProjection(cameras[0], uv0, rayStartPos0, rayDir0);
						vec3 world0 = cameras[0].pos + rayDir0 * c0.w;

						if (frame % 240 == 0 && length(uv1-vec2(0.5,0.5)) < 0.25) {
							c0.rgb = vec3(0.5)+0.5*sin(world1*100.);
						}

						float alpha = 0.95;
						if (frame == 0) alpha = 0.;
						if (c1.w == 1e10) alpha = 0.;
						if (c1.w != 1e10 && length(world0 - world1) > 1e-1) alpha = 0.;

						vec3 c = alpha * c0.xyz + (1 - alpha) * c1.xyz;
						//if (c1.w != 1e10 && length(world0 - world1) > 2e-1) c=vec3(1.,0.,0.);
						//c = vec3(0.5) + 0.5*sin((world0 - world1)*100.);
						imageStore(abuffer_image, ivec3(gl_FragCoord.xy, 1 - abuffer_read_layer), vec4(c, c1.w));
						col = vec4(c, 1.);
						//col = vec4(vec3(.5)+.5*sin(50*vec3(c1.w)), 1.0);
					}
				)
			);

		glUseProgram(draw);

		bindTexture("gbuffer", gbuffer);
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
		font.drawText(L"⏱: " + std::to_wstring(end-start), 10.f, 10.f, 15.f); // text, x, y, font size

		// this actually displays the rendered image
		swapBuffers();

		std::swap(cameras[0], cameras[1]);
		frame++;
	}
	return 0;
}
