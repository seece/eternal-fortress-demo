
#include "testbench.h"

const int screenw = 1024, screenh = 1024;

int main() {

	// create window and context. can also do fullscreen and non-visible window for compute only
	OpenGL context(screenw, screenh, "raymarcher");
	
	// load a font to draw text with -- any system font or local .ttf file should work
	Font font(L"Consolas");

	// shader variables; could also initialize them here, but it's often a good idea to
	// do that at the callsite (so input/output declarations are close to the bind code)
	Program march, draw;

	// this variable simply makes a scope-safe GL object so glCreateTexture is called here
	// and glDeleteTexture on scope exit; different types exist. gl_helpers.h contains more
	// helpers like this for buffers, renderbuffers and framebuffers.
	Texture<GL_TEXTURE_2D_ARRAY> state;

	glTextureStorage3D(state, 1, GL_RGBA32F, screenw, screenh, 3);

	// we're rendering from one image to the other and then back in a 'ping-pong' fashion; source keeps track of 
	// which image is the source and which is the target.
	int source_target = 0, frame = 0;

	while (loop()) // loop() stops if esc pressed or window closed
	{
		
		// timestamp objects make gl queries at those locations; you can substract them to get the time
		TimeStamp start;

		if (!march)
			// single-argument createProgram() makes a compute shader.
			// the argument can be either a filepath or the source directly as given by the GLSL macro:
			march = createProgram(
				GLSL(460,
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

				// uniform variables are global from the glsl perspective; you set them in the CPU side and every thread gets the same value
				uniform int source;
				uniform int frame;
				// images are also uniforms; we also need to declare the type of the image, here it's two channels of 32-bit floating point numbers
				layout(rgba32f) uniform image2DArray state;

				vec2 getThreadUV(uvec3 id) {
					return vec2(id.xy) / 1024.0;
				}

				void main() {
					// seed with seeds that change at different time offsets (not crucial to the algorithm but yields nicer results)
					srand(1u, uint(gl_GlobalInvocationID.x / 4) + 1u, uint(gl_GlobalInvocationID.y / 4) + 1u);
					srand(uint((rand())) + uint(frame/100), uint(gl_GlobalInvocationID.x / 4) + 1u, uint(gl_GlobalInvocationID.y / 4) + 1u);

					vec2 uv = getThreadUV(gl_GlobalInvocationID);
					vec4 color = vec4(uv, 0., 1.0);
					imageStore(state, ivec3(gl_GlobalInvocationID.xy, 1 - source), color);
				}
				));

		glUseProgram(march);
	
		// for convenience, uniforms can be bound directly with names
		glUniform1i("frame", frame);
		glUniform1i("source", source_target);
		// .. this includes images
		bindImage("state", 0, state, GL_READ_WRITE, GL_RGBA32F);
		// the arguments of dispatch are the numbers of thread blocks in each direction;
		// since our local size is 16x16x1, we'll get 1024x1024x1 threads total, just enough
		// for our image
		glDispatchCompute(64, 64, 1);

		// we're writing to an image in a shader, so we should have a barrier to ensure the writes finish
		// before the next shader call (wasn't an issue on my hardware in this case, but you should always make sure
		// to place the correct barriers when writing from compute shaders and reading in subsequent shaders)
		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

		// flip the ping-pongt flag
		source_target = 1 - source_target;

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
					uniform sampler2DArray state;
					out vec4 col;
					uniform int layer;
					void main() {
						col = vec4(texelFetch(state, ivec3(gl_FragCoord.xy, layer), 0).xyz, 1.0);
						//col = vec4(1., 0., 0., 1.);
						// other color schemes:
						//col = pow(texelFetch(state, ivec3(gl_FragCoord.xy, layer), 0).xyyx * vec3(.01, .0015, .002).zxyx, vec4(.8)) - vec4(.02);
						//col = vec4(texelFetch(state, ivec3(gl_FragCoord.xy, layer), 0).x*.005); // this rooughly matches figures from the aforementioned article
					}
				)
			);

		glUseProgram(draw);
		// textures are also a oneliner to bind (same for buffers; no bindings have to be states in the shader!)
		// note that this relies on the shader loader setting their values beforehand; if you use your own shader
		// loader, you'll have to replicate this (see how assignUnits() is used in program.cpp)
		bindTexture("state", state);
		glUniform1i("layer", source_target);
		glDrawArrays(GL_TRIANGLES, 0, 3);

		// here we're just using two timestamps, but you could of course measure intermediate timings as well
		TimeStamp end;

		// print the timing (word of warning; this forces a cpu-gpu synchronization)
		font.drawText(L"⏱: " + std::to_wstring(end-start), 10.f, 10.f, 15.f); // text, x, y, font size

		// this actually displays the rendered image
		swapBuffers();
		frame++;
	}
	return 0;
}
