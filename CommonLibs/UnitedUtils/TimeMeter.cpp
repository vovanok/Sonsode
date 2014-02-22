#include "TimeMeter.h"

TimeMeter::TimeMeter() {
	this->beginTime = 0;
	this->endTime = 0;
}

void TimeMeter::Begin() {
	LARGE_INTEGER time;
	QueryPerformanceCounter(&time);
	beginTime = time.QuadPart;
}

void TimeMeter::End() {
	LARGE_INTEGER time;
	QueryPerformanceCounter(&time);
	endTime = time.QuadPart;
}

double TimeMeter::Result() {
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	return ((double)abs(beginTime - endTime)) / (double)freq.QuadPart;
}