#pragma once

#include <vector>
#include "math_helpers.h"

struct CameraParameters {
	vec3 pos;
	float padding0;
	vec3 dir;
	float nearplane;
	vec3 up;
	float aspect;
	vec3 right;
	float padding3;
};

struct CameraPose {
	vec3 pos;
	vec3 dir;
	float zoom;
	std::string name;
};

static std::vector<CameraPose> loadPoses()
{
	std::vector<CameraPose> newPoses;
	FILE* fp = fopen("assets/cams.txt", "r");
	if (fp) {
		int num = 10;
		while (num >= 5) {
			CameraPose p = {};
			float theta = 0.f, phi = 0.f;
			char name[128] = { '\0' };
			num = fscanf(fp, "%f %f %f %f %f %f %127s\n",
				&p.pos.x,
				&p.pos.y,
				&p.pos.z,
				&phi,
				&theta,
				&p.zoom,
				name
			);
			if (num >= 5) {
				phi += 3.1415926536 / 2.f;
				p.dir = vec3(cos(theta)*sin(phi), sin(theta)*sin(phi), cos(phi));
			}
			if (num == 6) {
				p.name = std::string(name);
			}
			if (num >= 5) {
				newPoses.push_back(p);
				printf("loaded %s, phi: %f, theta: %f\n", name, phi, theta);
			}

		}
		fclose(fp);
	}
	return newPoses;
}



extern const int screenw;
extern const int screenh;

static void makeCamera(const CameraPose& pose, CameraParameters& cam)
{
	cam.pos = pose.pos;
	cam.dir = pose.dir;
	cam.right = normalize(cross(cam.dir, vec3(0.f, 1.f, 0.f)));
	cam.up = cross(cam.dir, cam.right);

	cam.aspect = float(screenw) / float(screenh);

	float nearplane = 0.1f;
	cam.dir *= nearplane;
	cam.right *= nearplane;
	cam.right /= pose.zoom;
	cam.up *= nearplane;
	cam.up /= pose.zoom;

	cam.nearplane = length(cam.dir);
}
