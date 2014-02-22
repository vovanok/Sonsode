#include "IterativeModel.h"

void IterativeModel::GpuOn() throw(std::string) {
	
	if (isGpuOn())
		return;

	if (Sonsode::GpuDeviceFactory::GpuDevices().empty()) {
		_isGpuOn = false;
		throw "Is not allowed GPU devices";
	}

	try {
		int orderNum = 0;
		for (auto &gpu : Sonsode::GpuDeviceFactory::GpuDevices())
			PrepareDataForGpu(*gpu, orderNum++);
	} catch(std::string e) {
		_isGpuOn = false;
		throw e;
	}

	_isGpuOn = true;
}

void IterativeModel::GpuOff() throw(std::string) {
	if (!isGpuOn())
		return;

	try {
		FreeDataForGpus();
	} catch(std::string e) {
		_isGpuOn = false;
		throw e;
	}

	_isGpuOn = false;
};
	
void IterativeModel::NextIteration(std::string methodName) throw(std::string) {
	double time;
	size_t iterNumber;

	NextIteration(methodName, time, iterNumber);
}

void IterativeModel::NextIteration(std::string methodName, double& time, size_t& iterNumber) throw(std::string) {
	if (CalculationMethods.count(methodName) <= 0)
		throw "Calculation method not found";

	tm.Begin();
	CalculationMethods[methodName]();
	tm.End();

	time = tm.Result();
	iterNumber = currentIteration();
		
	_currentIteration++;
	_currentTime += tau();
};