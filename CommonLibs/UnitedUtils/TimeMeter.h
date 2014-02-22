#ifndef TIMEMETER_H
#define TIMEMETER_H

#include <Windows.h>
#include <math.h>

class TimeMeter {

protected:
	LONGLONG beginTime;
	LONGLONG endTime;

public:
	TimeMeter();
	void Begin();
	void End();
	double Result();
};

#endif