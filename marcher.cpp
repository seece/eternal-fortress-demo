
#include "testbench.h"
#include "mp3music.h"
#include "cameras.h"
#include <cinttypes>
#include <cassert>
#include <deque>
#include <chrono>

const int screenw = 1280, screenh = 720;
constexpr int MAX_POINT_COUNT = 10. * screenw * screenh;
static constexpr GLuint SAMPLE_BUFFER_TYPE = GL_RGBA16F;
static constexpr GLuint JITTER_BUFFER_TYPE = GL_RG8;
static bool showDebugInfo = false;

static void setWrapToClamp(GLuint tex) {
	glTextureParameteri(tex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTextureParameteri(tex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

struct RgbPoint {
	vec3 xyz;
	uint32_t normalSpecularSun;
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

// returns a vector in [-1, 1]^2
vec2 getRandomJitter()
{
	static int xi;
	static int yi;
	int xs[] = { 4, 5, 13, 16, 0, 13, 18, 6, 14, 9, 15, 16, 9, 10, 17, 17, 6, 10, 18, 9, 9, 11, 15, 7, 8, 12, 19, 9, 0 };
	int ys[] = { 1, 18, 9, 19, 4, 0, 14, 6, 1, 5, 9, 2, 2, 13, 7, 4, 13, 14, 12, 6, 7, 1, 10, 17, 5, 3, 6, 17, 20, 19, 13, 19, 2, 0, 12, 1, 20, 15, 6, 1, 7 };
	int x = xs[xi];
	int y = ys[yi];
	xi = (xi+1) % (sizeof(xs) / sizeof(int));
	yi = (yi+1) % (sizeof(ys) / sizeof(int));
	return vec2(x - 10, y - 10) / 20.f;
}


struct Shot {
	// static
	std::string name;
	std::string camName;
	float start = 0.f;
	// dynamic state
	float end =0.f;
	float length = 0.f;
	float relative = 0.f;
	float ratio = 0.f;
};

static std::vector<Shot> shots;
static std::map<std::string, Shot&> shotNames;
static std::vector<CameraPose> cameraPoses;
static std::map<std::string, CameraMove> cameraMoves;

static CameraPose* findPose(const std::string& name) {
	for (CameraPose& p : cameraPoses) {
		if (p.name == name)
			return &p;
	}
	return nullptr;
}

static std::vector<Shot> loadShots()
{
	std::vector<Shot> news;
	FILE* fp = fopen("assets/shots.txt", "r");
	if (fp) {
		int num = 10;
		int idx = 0;
		while (num >= 2) {
			Shot s = {};
			char camname[128] = { '\0' };
			char shotname[128] = { '\0' };
			num = fscanf(fp, "%f %127s\n",
				&s.start,
				camname
			);
			if (num >= 2) {
				s.camName = camname;
				//printf("loaded shot %s\n", name);
			}
			if (num >= 3) {
				s.name = shotname;
			}
			else {
				s.name = "shot" + std::to_string(idx);
			}
			if (num >= 2) {
				news.push_back(s);
				idx++;
			}

		}
		fclose(fp);
	}
	return news;
}

Shot shotAtTime(float secs) {
	Shot s = {};
	for (int i = 0; i < shots.size() - 1; i++) {
		if (shots[i + 1].start > secs) {
			s = shots[i];
			s.end = shots[i + 1].start;
			s.length = s.end - shots[i].start;
			s.relative = secs - shots[i].start;
			s.ratio = s.relative / s.length;
			return s;
		}
	}

	return shots[shots.size() - 1];
}

static void cameraPath(const Shot& shot, CameraPose& outPose)
{
	CameraPose pose = {};
	int poseid = 0;
	for (int i = 0; i < cameraPoses.size(); i++) {
		if (cameraPoses[i].name == shot.camName) {
			pose = cameraPoses[i];
			poseid = i;
			break;
		}
		if (i == cameraPoses.size() - 1) printf("Error! No camera found: %s\n", shot.camName.c_str());
	}

	CameraMove move = {};
	if (cameraMoves.find(shot.camName) != cameraMoves.end()) {
		move = cameraMoves[shot.camName];
	}

	float t = shot.relative;
	float tuniq = shot.relative + poseid * 31.3;
	pose.pos += t * move.axis;
	pose.pos += t * pose.dir * move.forward;
	float tt = t * 0.1f;
	//pose.pos = vec3(0. + 2.0 * sin(tt), -4., 7.f + 0.1f*cos(tt));
	//pose.dir = normalize(vec3(0.9f*cos(tt*0.5f), 0.3 + 0.9f*sin(tt), -1.f));
	//pose.zoom = 0.5f;

	pose.dir += move.shake * 0.01f*vec3(
		pow(cos(tuniq*0.8), 3.0f),
		0.6f*pow(sin(tuniq*1.4f), 3.0f),
		pow(sin(tuniq*1.0f + sin(tuniq)), 1.0f));
	pose.dir = normalize(pose.dir);

	outPose = pose;
}

static void reloadAnimations(Music& music)
{
	std::vector<CameraPose> newPoses = loadPoses();
	if (newPoses.size() > 0) {
		cameraPoses = newPoses;
	}
	std::vector<Shot> newShots = loadShots();
	if (newShots.size() > 0) {
		shots = newShots;
		Shot s;
		s.start = music.getDuration();
		s.camName = "floaters";
		shots.push_back(s);
	}
	std::map<std::string, CameraMove> newMoves = loadMoves();
	if (newMoves.size() > 0) {
		cameraMoves = newMoves;
	}
}

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
	//Texture<GL_TEXTURE_2D> skyboxTexture = loadImage(L"assets/MonValley_A_LookoutPoint_8k.jpg");
	Texture<GL_TEXTURE_2D> dayIrradiance = loadImage(L"assets/champagne/iem.png");
	std::wstring cubePaths[] = {
		L"assets/champagne/_posx.png",
		L"assets/champagne/_negx.png",
		L"assets/champagne/_negy.png",
		L"assets/champagne/_posy.png",
		L"assets/champagne/_posz.png",
		L"assets/champagne/_negz.png"
	};
	Texture<GL_TEXTURE_CUBE_MAP> skyboxCubemap = loadCubeMap(cubePaths, sizeof(cubePaths) / sizeof(std::wstring), true);

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
	Buffer rayIndexBuffer;

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

	int jumpBufferMaxElements = dim2nodecount(max(screenw, screenh));
	int rayIndexBufferMaxElements = 0;
	int maxdim = 1 << (int(log2(max(screenw, screenh))) + 1);
	vec2 screenBoundary(screenw / float(maxdim), screenh / float(maxdim));

	// Build index array for the raymarcher
    {
		std::vector<int> indexArray;

        vec2 scale = (float(maxdim) / float(screenw), float(maxdim) / float(screenw));
		float aspect = float(screenw) / float(screenh);

        for (int i = 0; i < jumpBufferMaxElements; i++) {
            int b = tobin(i);
            int start = binto(b);
            int z = i - start;
            uvec2 coord = z2xy(uint(z));
            int idim = 1 << b;
            int size = idim * idim;
            float dim = float(idim);

            vec2 uv = vec2(0.5/dim) + vec2(coord) / vec2(dim);
			//float margin = 0.5f / dim; // no effect?
			float margin = 0.f;
			if ( uv.x <= screenBoundary.x + margin && uv.y <= screenBoundary.y + margin) {
				indexArray.push_back(i);
			}
        }
		rayIndexBufferMaxElements = indexArray.size();
		printf("rayIndexBuffer elements: %d\n", rayIndexBufferMaxElements);
		glNamedBufferStorage(rayIndexBuffer, indexArray.size() * sizeof(int), indexArray.data(), 0);
    }

	glNamedBufferStorage(jumpbuffer, jumpBufferMaxElements * sizeof(float), NULL, 0);
	glNamedBufferStorage(radiusbuffer, jumpBufferMaxElements * sizeof(float), NULL, 0);
	glNamedBufferStorage(uvbuffer, 2 * jumpBufferMaxElements * sizeof(float), NULL, 0);

	GLint64 jumpBufferSize = -1;
	glGetNamedBufferParameteri64v(jumpbuffer, GL_BUFFER_SIZE, &jumpBufferSize);
	printf("jumpBuffer size: %" PRId64 " bytes = %.3f MiB\n", jumpBufferSize, jumpBufferSize / 1024. / 1024.);
	
	Music music(L"assets/final3_fraktals.wav");
	reloadAnimations(music);
	music.play();

	bool interactive = false;
	bool controls = true;
	std::deque<double> frameTimes;
	auto lastFrameTime = std::chrono::high_resolution_clock::now();

	while (loop()) // loop() stops if esc pressed or window closed
	{
		TimeStamp start;
		float secs = music.getTime();
		double dt = 0.;
		{
			auto now = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now - lastFrameTime).count();
			double dtnew = duration / 1e9;
			//if (dtnew > 1.) dtnew = 1. / 30.;
			if (frameTimes.size() >= 6) {
				frameTimes.pop_front();
			}
			frameTimes.push_back(dtnew);
			for (double ft : frameTimes) { dt += ft; }
			dt /= frameTimes.size();
			lastFrameTime = std::chrono::high_resolution_clock::now();
		}


		if (frame == 0 || (controls && (frame % 4 == 0))) {
			reloadAnimations(music);
		}

		float futureInterval = dt * 4;

		Shot futureShot = shotAtTime(secs + futureInterval);
		Shot currentShot = shotAtTime(secs);

		CameraPose pose, futurePose;
		if (interactive) {
			pose = cameraPoses[0];
			futurePose = pose;
		}
		else {
			cameraPath(currentShot, pose);
			cameraPath(futureShot, futurePose);
		}
		makeCamera(futurePose, cameras[1]);
		glNamedBufferSubData(cameraData, 0, sizeof(cameras), &cameras);
		vec3 sunDirection = normalize(vec3(-0.5f, -1.0f, 0.7f));
		vec3 sunColor = vec3(1., 0.8, 0.5);

		if (controls) {
			float seekTime = 1.f;
			if (keyDown(VK_LSHIFT) || keyDown(VK_RSHIFT)) {
				seekTime *= 5.f;
			}
			if (keyDown(VK_LEFT)) {
				music.seek(music.getTime() - seekTime);
			}
			else if (keyDown(VK_RIGHT)) {
				music.seek(music.getTime() + seekTime);
			}
			if (keyHit(VK_SPACE)) music.togglePlaying();
			if (keyHit(VK_BACK)) music.seek(0.);
			if (keyHit(0x4D)) music.setVolume(music.getVolume() > -100. ? -100. : 0.);
			if (keyHit(VK_F2)) interactive = !interactive;
			if (keyHit(VK_RETURN)) {
				music.seek(currentShot.start);
			}
		}

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

		vec2 cameraJitter = getRandomJitter() / float(max(screenw, screenh));

		glUniform1i("frame", frame);
		glUniform1f("secs", secs);
		glUniform2i("screenSize", screenw, screenh);
		glUniform2f("screenBoundary", screenBoundary.x, screenBoundary.y);
		glUniform2f("cameraJitter", cameraJitter.x, cameraJitter.y);
		glUniform3f("sunDirection", sunDirection.x, sunDirection.y, sunDirection.z);
		glUniform3f("sunColor", sunColor.x, sunColor.y, sunColor.z);
		glUniform1i("pointBufferMaxElements", pointBufferMaxElements);
		glUniform1i("jumpBufferMaxElements", jumpBufferMaxElements);
		glUniform1i("rayIndexBufferMaxElements", rayIndexBufferMaxElements);
		bindBuffer("cameraArray", cameraData);
		bindBuffer("pointBufferHeader", pointBufferHeader);
		bindBuffer("pointBuffer", pointBuffer);
		bindBuffer("jumpBuffer", jumpbuffer);
		bindBuffer("rayIndexBuffer", rayIndexBuffer);
		bindBuffer("uvBuffer", uvbuffer);
		bindBuffer("radiusBuffer", radiusbuffer);
		bindBuffer("debugBuffer", debugBuffer);
		bindBuffer("stepBuffer", stepBuffer);
		glUniform3i("noiseOffset", rand() % 64, rand() % 64, noiseLayer);
		bindTexture("noiseTextures", noiseTextures);
		bindTexture("skybox", skyboxCubemap);
		bindTexture("skyIrradiance", dayIrradiance);
		//bindImage("gbuffer", 0, gbuffer, GL_WRITE_ONLY, GL_RGBA32F);
		bindImage("zbuffer", 0, zbuffer, GL_WRITE_ONLY, GL_R32F);
		bindImage("edgebuffer", 0, edgebuffer, GL_WRITE_ONLY, GL_R8);
		bindImage("samplebuffer", 0, samplebuffer, GL_WRITE_ONLY, SAMPLE_BUFFER_TYPE);
		bindImage("jitterbuffer", 0, jitterbuffer, GL_WRITE_ONLY, JITTER_BUFFER_TYPE);

		glDispatchCompute(2 * screenw/16, 2 * screenh/16, 1); // TODO scale to max GPU occupancy?

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
				float aspect;
				vec3 right;
				float padding3;
			};

            struct RgbPoint {
                vec3 xyz;
                uint normalSpecularSun;
                vec4 rgba;
            };

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

			layout(r8) uniform image2D edgebuffer;
			uniform int pointBufferMaxElements;
			uniform int numberOfPointsToSplat;
			uniform ivec2 screenSize;
			uniform ivec3 noiseOffset;
            uniform vec3 sunDirection;
            uniform vec3 sunColor;
            uniform samplerCube skybox;
			uniform sampler2DArray noiseTextures;

            vec3 getNoise(ivec2 coord, int ofs = 0)
            {
                return texelFetch(noiseTextures,
                        ivec3((coord.x + noiseOffset.x) % 64, (coord.y + noiseOffset.y) % 64, (noiseOffset.z + ofs) % 64), 0).rgb;
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

			vec3 projectPoint(CameraParams cam, vec3 p, out vec3 fromCamToPoint) {
				vec3 op = p - cam.pos;
                fromCamToPoint = op;
				float n = length(cam.dir);
				float z = dot(cam.dir, op) / n;
				vec3 pp = (op * n) / z;
                vec3 up = cam.up / cam.aspect;
				vec2 plane = vec2(
					dot(pp, cam.right) / dot(cam.right, cam.right),
					dot(pp, up) / dot(up, up)
				);
				return vec3(plane + vec2(0.5, 0.5 * cam.aspect), z);
			}

            // https://software.intel.com/en-us/node/503873
            vec3 RGB_YCoCg(vec3 c)
            {
                return vec3(
                        c.x / 4.0 + c.y / 2.0 + c.z / 4.0,
                        c.x / 2.0 - c.z / 2.0,
                        -c.x / 4.0 + c.y / 2.0 - c.z / 4.0
                        );
            }

            int sampleNum = 0;

			void addRGB(uint pixelIdx, vec3 c, ivec2 pix) {
                vec3 scale = vec3(1000.);
                //c += (vec3(0.5)/scale)*(getNoise(pix, sampleNum++) - vec3(.55));
                c = max(vec3(0.), c) * scale;
                uint redblue = ((uint(c.b) << 16) & 0xffff0000) | (uint(c.r) & 0xffff);
                uint green = uint(c.g) & 0xffff;
                atomicAdd(colors[pixelIdx*2],   redblue);
                atomicAdd(colors[pixelIdx*2+1], green);
			}

            vec3 applyFog( in vec3  rgb,      // original color of the pixel
                    in float distance, // camera to point distance
                    in vec3  rayOri,   // camera position
                    in vec3  rayDir )  // camera to point vector
            {
                float b = 0.01;
                float c = 1.0;
                float fogAmount = c * exp(-(rayOri.y)*b) * (1.0-exp( -distance*rayDir.y*b ))/rayDir.y;
                fogAmount = clamp(fogAmount, 0., 1.);
                float scatter = pow(max(0., dot(rayDir, sunDirection)), 10.);
                vec3  fogColor = mix(vec3(0.5,0.6,0.7), vec3(1., 0.8, 0.), scatter);

                return mix( rgb, fogColor, fogAmount );
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
				vec3 pos = points[index].xyz;
				vec4 color = points[index].rgba;
				vec3 c = color.rgb;

				// Raymarcher never produces pure (0, 0, 0) hits.
				if (pos == vec3(0.))
					return;

                vec3 fromCamToPoint;
				vec3 camSpace = projectPoint(cameras[1], pos, fromCamToPoint);
				vec2 screenSpace = camSpace.xy * vec2(screenSize.x, screenSize.y);

				int x = int(screenSpace.x);
				int y = int(screenSpace.y);

				if (x < 0 || y < 0 || x >= screenSize.x || y >= screenSize.y)
					return;

                vec4 normalSpecularSun = unpackUnorm4x8(points[index].normalSpecularSun);
                vec3 normal = decodeNormal(normalSpecularSun.xy);
                float materialShininess = normalSpecularSun.z;
                float sun = normalSpecularSun.w;

				int pixelIdx = screenSize.x * y + x;
				bool isEdge = imageLoad(edgebuffer, ivec2(x, y)).x > 0;

                vec3 toCamera = normalize(-fromCamToPoint);
                vec3 H = normalize(sunDirection + toCamera);
                float specular = pow(max(0., dot(normal, H)), 10.);
                c = mix(c, (specular * sun) * c, materialShininess);
                //c = vec3(specular * sun);

				float distance = length(fromCamToPoint);
				//float fog = pow(min(1., distance / 10.), 1.0);
				//c = mix(c, vec3(0.1, 0.1, 0.2)*0.1, fog);

                c = applyFog( c,      // original color of the pixel
                    distance, // camera to point distance
                    cameras[1].pos,   // camera position
                    -toCamera);  // camera to point vector


                //c = vec3(.5) + .5*sin(pos*1.4);

				c = c / (vec3(1.) + c);
				c = clamp(c, vec3(0.), vec3(1.));

				//float weight = max(0.1, min(1e3, 1. / (pow(camSpace.z / 3., 2.) + 0.001)));
				float weight = max(0.1, min(1e1, 1. / (pow(camSpace.z / 3., 2.) + 0.001)));
                const int weight_scale = 500;

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
                addRGB(idx,                     ws[0] * col, ivec2(x, y));
                addRGB(idx + 1,                 ws[1] * col, ivec2(x+1, y));
                addRGB(idx + screenSize.x,      ws[2] * col, ivec2(x, y+1));
                addRGB(idx + screenSize.x + 1,  ws[3] * col, ivec2(x+1, y+1));

                atomicAdd(sampleWeights[idx], (uint(weight_scale * weight * ws[0]) << 14) | uint(255 * ws[0]));
                atomicAdd(sampleWeights[idx + 1], (uint(weight_scale * weight * ws[1]) << 14) | uint(255 * ws[1]));
                atomicAdd(sampleWeights[idx + screenSize.x], (uint(weight_scale * weight * ws[2]) << 14) | uint(255 * ws[2]));
                atomicAdd(sampleWeights[idx + screenSize.x + 1], (uint(weight_scale * weight * ws[3]) << 14) | uint(255 * ws[3]));

				atomicAdd(pointsSplatted, 1); // TODO DEBUG HACK REMOVE!
			}
			));
		}

		int numberOfPointsToSplat = MAX_POINT_COUNT;

		makeCamera(pose, cameras[1]);
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
		glUniform3f("sunDirection", sunDirection.x, sunDirection.y, sunDirection.z);
		glUniform3f("sunColor", sunColor.x, sunColor.y, sunColor.z);
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

		if (false) {
			printf("debugData: i: %d, parent: %d, size: %d, b: %d, start: %d, parent_size: %d,\nradius: %f, zdepth: %f, parentDepth: %f\n",
				debugData.i, debugData.parent, debugData.size, debugData.b, debugData.start,
				debugData.parent_size,
				debugData.pixelRadius, debugData.zdepth, debugData.parentDepth);
			printf("nearPlane: %f, projPlaneDist: %f\n", debugData.nearPlane, debugData.projPlaneDist);
			printf("parent UV: (%f, %f), child UV: (%f, %f)\n",
				debugData.parentUVx, debugData.parentUVy, debugData.childUVx, debugData.childUVy);
		}

		if (showDebugInfo) {
			int data[2];
			glGetNamedBufferSubData(pointBufferHeader, 0, 8, data);
			//printf("currentWriteOffset: %d\n", data[0]);
			pointsSplatted = data[1];
			//printf("pointsSplatted: %d\t(%.3f million)\n", data[1], data[1]/1000000.);
		}
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
						float aspect;
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
                    uniform samplerCube skybox;
                    uniform vec3 sunDirection;
                    uniform vec3 sunColor;

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

					vec3 getNoise(ivec2 coord, int ofs = 0)
					{
						return texelFetch(noiseTextures,
							ivec3((coord.x + noiseOffset.x) % 64, (coord.y + noiseOffset.y) % 64, (noiseOffset.z + ofs) % 64), 0).rgb;
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
                        vec3 c = texture(skybox, dir).rgb;
                        c = 3.*pow(c, vec3(1.5));

                        float d = dot(dir, sunDirection);
                        /*
                        float horizon = dir.y;
                        vec3 base = max(0., horizon) * vec3(0., 0.01, 0.03)
                                    + max(0., -horizon) * .1 * vec3(0.6, 0.0, 1.0)
                                    + max(0., pow(1.-abs(horizon), 16.)) * vec3(0.1, 0.00, 0.)
                                    ;
                        float disc = 50. * pow(max(0., (d)), 1000.);
                        float shine =  pow(max(0., d*.8), 20.);
                        //return vec3(disc);
                        //return c * (base + vec3(disc) + shine * sunColor);
                        */
                        float ground = pow(max(0., 1.*dir.y), 10.);
                        float sky = pow(-min(0., dir.y)+0.5, 1.);
                        float horizon = pow(1-abs(dir.y+0.2), 20.);
                        //return mix(c.bgr, 0.005*vec3(0.3, 0.3, 0.5), ground);
                        //return mix(vec3(0., 1., 0.), c.bgr, sky);
                        return c.bgr;
                    }

                    void main() {
                        int pixelIdx = screenSize.x * int(gl_FragCoord.y) + int(gl_FragCoord.x);
                        uvec3 icolor = uvec3(
                                colors[3 * pixelIdx + 0],
                                colors[3 * pixelIdx + 1],
                                colors[3 * pixelIdx + 2]
                                );

                        uint weightAlphaPacked = sampleWeights[pixelIdx];
                        uint fixedWeight = weightAlphaPacked >> 14;
                        uint fixedAlpha = weightAlphaPacked & 0x3fff;
                        float weight = float(fixedWeight) / 500.;
                        float alpha = float(fixedAlpha) / 255.;
                        alpha += weight*.1;

                        alpha = pow(min(1., alpha/2), 0.5);

                        uint packedColor1 = colors[pixelIdx*2];
                        uint packedColor2 = colors[pixelIdx*2+1];
                        vec3 color = vec3(
                                packedColor1 & 0xffff,
                                packedColor2 & 0xffff,
                                (packedColor1 & 0xffff0000) >> 16);
                        color /= vec3(1000.);
                        //color = pow(color, vec3(1./0.7));

                        // "color" is now Reinhard tone mapped

                        if (weight > 0.) {
                            color /= weight;
                        } else {
                            color = vec3(0.);
                        }

                        //if (weight < 4.) color=vec3(1., 0., 0.);
                        //color = vec3(weight>10.);

                        vec3 p, dir;
                        vec2 uv = vec2(gl_FragCoord.x, gl_FragCoord.y) / screenSize;
                        uv.y /= cameras[1].aspect;
                        getCameraProjection(cameras[1], uv, p, dir);

                        vec3 skyColor = sampleSky(dir);
                        vec3 c = mix(skyColor, color, alpha);

                        vec3 srgb = linearToSRGB(c.rgb);
                        vec3 noise = 1./255. * getNoise(ivec2(gl_FragCoord.xy));
                        srgb += noise;
                        outColor = vec4(srgb, 1.);

                        // Clear the accumulation buffer
                        colors[pixelIdx*2] = 0;
                        colors[pixelIdx*2+1] = 0;
                        sampleWeights[pixelIdx] = 0;
                    }
				)
			);

		glUseProgram(draw);

		glUniform1i("frame", frame);
		glUniform3i("noiseOffset", rand() % 64, rand() % 64, noiseLayer);
		glUniform1f("secs", secs);
		glUniform2i("screenSize", screenw, screenh);
		glUniform3f("sunDirection", sunDirection.x, sunDirection.y, sunDirection.z);
		glUniform3f("sunColor", sunColor.x, sunColor.y, sunColor.z);
		bindTexture("noiseTextures", noiseTextures);
		bindTexture("skybox", skyboxCubemap);
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

		if (controls) {
			// print the timing (word of warning; this forces a cpu-gpu synchronization)
			font.drawText(L"Total: " + std::to_wstring(end - start), 10.f, 10.f, 15.f); // text, x, y, font size
			font.drawText(L"Draw: " + std::to_wstring(drawTime - start), 10.f, 25.f, 15.f);
			font.drawText(L"Splat: " + std::to_wstring(splatTime - drawTime), 10.f, 40.f, 15.f);
			font.drawText(L"PostProc: " + std::to_wstring(end - splatTime), 10.f, 55.f, 15.f);
			font.drawText(L"Points: " + std::to_wstring(pointsSplatted / 1000. / 1000.) + L" M", 200.f, 10.f, 15.f);
			font.drawText(L"Music: " + std::to_wstring(music.getTime()) + L" s", 200.f, 25.f, 15.f);
			{
				std::wstring ws, ws2;
				ws.assign(currentShot.camName.begin(), currentShot.camName.end());
				ws2.assign(currentShot.name.begin(), currentShot.name.end());
				font.drawText(L"Shot/Cam: " + ws + L"/" + ws2 + L" @ " + std::to_wstring(currentShot.relative) + L" s", 200.f, 40.f, 15.f);
			}
			font.drawText(interactive ? L"Interactive pose" : L"Cam track pose", 200.f, 55.f, 15.f);
			font.drawText(L"dt: " + std::to_wstring(dt) + L" s", 200.f, 70.f, 15.f);
			font.drawText(std::to_wstring(1./dt) + L" Hz", 400.f, 70.f, 15.f);
		}

		// this actually displays the rendered image
		swapBuffers();

		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

		std::swap(cameras[0], cameras[1]);
		frame++;
	}
	return 0;
}
