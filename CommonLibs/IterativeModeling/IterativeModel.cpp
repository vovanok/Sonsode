#include "IterativeModel.h"

void IterativeModel::GpuOn() {
	if (isGpuOn())
		return;

	if (Sonsode::GpuDeviceFactory::GpuDevices().empty()) {
		_isGpuOn = false;
		throw std::exception("Is not allowed GPU devices");
	}

	try {
		int orderNum = 0;
		for (auto &gpu : Sonsode::GpuDeviceFactory::GpuDevices())
			PrepareDataForGpu(*gpu, orderNum++);
	} catch(std::exception e) {
		_isGpuOn = false;
		throw e;
	}

	_isGpuOn = true;
}

void IterativeModel::GpuOff() {
	if (!isGpuOn())
		return;

	try {
		FreeDataForGpus();
	} catch(std::exception e) {
		_isGpuOn = false;
		throw e;
	}

	_isGpuOn = false;
};
	
void IterativeModel::NextIteration(std::string methodName) {
	double time;
	size_t iterNumber;

	NextIteration(methodName, time, iterNumber);
}

void IterativeModel::NextIteration(std::string methodName, double& time, size_t& iterNumber) {
	if (CalculationMethods.count(methodName) <= 0)
		throw std::exception("Calculation method not found");

	tm.Begin();
	CalculationMethods[methodName]();
	tm.End();

	time = tm.Result();
	iterNumber = currentIteration();
		
	_currentIteration++;
	_currentTime += tau();
};