
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
	Texture<GL_TEXTURE_2D_ARRAY> abuffer;
	Texture<GL_TEXTURE_2D> gbuffer;

	int renderw = screenw, renderh = screenh;

	glTextureStorage3D(abuffer, 1, GL_RGBA32F, screenw, screenh, 2);
	glTextureStorage2D(gbuffer, 1, GL_RGBA32F, renderw, renderh);

	glTextureStorage3D(state, 1, GL_RGBA32F, screenw, screenh, 3);

	// we're rendering from one image to the other and then back in a 'ping-pong' fashion; source keeps track of 
	// which image is the source and which is the target.
	int abuffer_read_layer = 0, frame = 0;

	while (loop()) // loop() stops if esc pressed or window closed
	{
		
		// timestamp objects make gl queries at those locations; you can substract them to get the time
		TimeStamp start;

		if (!march)
			march = createProgram("shaders/marcher.glsl");

		glUseProgram(march);
	
		glUniform1i("frame", frame);
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

		// flip the ping-pongt flag
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
					uniform sampler2D gbuffer;
					uniform sampler2DArray abuffer;
					layout(rgba32f) uniform image2DArray abuffer_image;
					out vec4 col;
					uniform int abuffer_read_layer;

					void main() {
						vec4 c0 = texelFetch(abuffer, ivec3(gl_FragCoord.xy, abuffer_read_layer), 0);
						vec4 c1 = texelFetch(gbuffer, ivec2(gl_FragCoord.xy), 0);

						float alpha = 0.8;
						vec3 c = alpha * c0.xyz + (1 - alpha) * c1.xyz;
						imageStore(abuffer_image, ivec3(gl_FragCoord.xy, 1 - abuffer_read_layer), vec4(c, c1.w));
						col = vec4(c, 1.);
						//col = vec4(vec3(.5)+.5*sin(50*vec3(c1.w)), 1.0);
					}
				)
			);

		glUseProgram(draw);
		// textures are also a oneliner to bind (same for buffers; no bindings have to be states in the shader!)
		// note that this relies on the shader loader setting their values beforehand; if you use your own shader
		// loader, you'll have to replicate this (see how assignUnits() is used in program.cpp)
		bindTexture("gbuffer", gbuffer);
		bindTexture("abuffer", abuffer);
		bindImage("abuffer_image", 0, abuffer, GL_WRITE_ONLY, GL_RGBA32F);
		glUniform1i("abuffer_read_layer", abuffer_read_layer);
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
