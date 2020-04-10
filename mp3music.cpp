#define WIN32_LEAN_AND_MEAN
#define WIN32_EXTRA_LEAN
#include <Windows.h>
#include <dshow.h>
#include "mp3music.h"

#pragma comment(lib, "strmiids")

Music::Music(LPCWSTR filepath)
{
	IPin * pin;
	IBaseFilter * src;
	//IBasicAudio * audiocontrol;
	long long lduration;
	IGraphBuilder * graphbuilder;

	CoInitialize(0);
	CoCreateInstance(CLSID_FilterGraph, 0, CLSCTX_INPROC, IID_IGraphBuilder, (void**)&graphbuilder);

	graphbuilder->QueryInterface(IID_IMediaControl, (void**)&mediacontrol);
	graphbuilder->QueryInterface(IID_IMediaSeeking, (void**)&mediaseeking);
	graphbuilder->QueryInterface(IID_IBasicAudio  , (void**)&audiocontrol);

	graphbuilder->AddSourceFilter(filepath, L"piisi", &src);
	src->FindPin(L"Output", &pin);

	graphbuilder->Render(pin);

	//audiocontrol->put_Volume((long)-200); 
	setVolume(-0.2); // only a small reduction to reduce clipping

	pin->Release();
	src->Release();
	//audiocontrol->Release();
	//graphbuilder->Release();

	mediaseeking->SetTimeFormat(&TIME_FORMAT_MEDIA_TIME);
	mediaseeking->GetDuration(&lduration);
	duration = (long double)lduration/(long double)10000000.0;
	long long pos = 0;
	mediaseeking->SetPositions(&pos, AM_SEEKING_AbsolutePositioning, &pos, AM_SEEKING_NoPositioning);

	playing = false;
}

Music::~Music()
{
	audiocontrol->Release();
	mediacontrol->Release();
	mediaseeking->Release();
}

int Music::play()
{
	mediacontrol->Run();
	playing = true;

	return 0;
}

int Music::pause()
{
	mediacontrol->Pause();
	playing = false;
	return 0;
}

void Music::togglePlaying() {
	if (playing) {
		pause();
	} else {
		play();
	}
}

int Music::seek(long double time)
{
	long long pos = __int64(10000000.0*min(duration, max(.0,time)));
	mediacontrol->Stop();
	mediaseeking->SetPositions(&pos, AM_SEEKING_AbsolutePositioning, &pos, AM_SEEKING_NoPositioning);
	if (playing) {
		mediacontrol->Run();
	}
	return 0;
}

double Music::getTime()
{
	long long res;
	mediaseeking->GetCurrentPosition((LONGLONG*)&res);
	return (double)(res)/(double)10000000.0;
}

int Music::setVolume(double decibels)
{
	audiocontrol->put_Volume(long(decibels * 100.0));

	return 0;
}

double Music::getVolume()
{
	long vol;
	audiocontrol->get_Volume(&vol);
	return double(vol)/100.0;
}

double Music::getDuration() {
	return duration;
}