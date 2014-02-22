#pragma once

#include "InitRoutines.h"

namespace CapacityTests {
	void HeatConductivity2D(size_t countIteration, size_t startDim, size_t finishDim, size_t stepDim, std::string methodName);
	void HeatConductivity3D(size_t countIteration, size_t startDim, size_t finishDim, size_t stepDim, std::string methodName);
	void OilSpillage(size_t countIteration, size_t startDim, size_t finishDim, size_t stepDim, std::string methodName);
	void AirFlow(size_t countIteration, size_t startDim, size_t finishDim, size_t stepDim, std::string methodName);
	void FireSpread();

	void FireSpreadRealAndModelTimeRelation(size_t maxModelingTimeSec, size_t modelingTimeStepSec,
			size_t startDim, size_t finishDim, size_t stepDim, std::string methodName);
	void OilSpillageRealAndModelTimeRelation(size_t maxModelingTimeSec, size_t modelingTimeStepSec,
			size_t startDim, size_t finishDim, size_t stepDim, std::string methodName);
}