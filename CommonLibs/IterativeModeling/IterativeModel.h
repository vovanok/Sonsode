#pragma once

#include <map>
#include <functional>
#include "GpuDevice.hpp"
#include "GpuDeviceFactory.h"
#include "TimeMeter.h"

class IterativeModel {
public:
	IterativeModel(float tau)
		: tm(TimeMeter()), _isGpuOn(false),
			_currentIteration(0), _currentTime(0), _tau(tau) { }
	virtual ~IterativeModel() { }
	
	void GpuOn() throw(std::string);
	void GpuOff() throw(std::string);
	
	void NextIteration(std::string methodName) throw(std::string);
	void NextIteration(std::string methodName, double& time, size_t& iterNumber) throw(std::string);
	
	std::map<std::string, std::function<void()>> CalculationMethods;

	void AddCalculationMethod(std::string methodName, std::function<void(void)> calculationMethod) {
		CalculationMethods[methodName] = calculationMethod;
	}

	void DelCalculationMethod(std::string methodName) {
		CalculationMethods.erase(methodName);
	}

	size_t currentIteration() const { return _currentIteration; }
	float currentTime() const { return _currentTime; }
	float tau() const { return _tau; }
	bool isGpuOn() const { return _isGpuOn; }

	virtual std::string PrintData() const = 0;
	virtual void SynchronizeWithGpu() = 0;
protected:
	TimeMeter tm;

	size_t _currentIteration;
	float _currentTime;
	float _tau;
	bool _isGpuOn;

	virtual void PrepareDataForGpu(const Sonsode::GpuDevice &gpuDevice, size_t orderNumber) throw(std::string) = 0;
	virtual void FreeDataForGpus() throw(std::string) = 0;
};