#pragma once

// Pauli's DirectShow-player class.

#ifndef MP3MUSIC_H
#define MP3MUSIC_H

#include <DShow.h>

class Music
{
private:
	double duration;
	IMediaControl * mediacontrol;
	IMediaSeeking * mediaseeking;
	IBasicAudio * audiocontrol;
	int release();
public:
	Music(LPCWSTR filepath);
	~Music();
	int play();
	int pause();
	void togglePlaying();
	int seek(long double time);
	// set volume takes the input as decibels. -100.0 is silence, 0.0 is full blast
	int setVolume(double decibels);
	// returns volume as decibels
	double getVolume();
	double getTime();
	bool playing;
	double getDuration();
};

#endif