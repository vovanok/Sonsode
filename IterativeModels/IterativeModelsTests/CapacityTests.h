#pragma once

#include "InitRoutines.h"

namespace CapacityTests {
	void ModelPerfomanceTest2D(std::function<IterativeModel*(size_t, size_t)> modelGetter, std::string methodName,
			size_t countIteration, size_t startDim, size_t finishDim, size_t stepDim);

	void ModelPerfomanceTest3D(std::function<IterativeModel*(size_t, size_t, size_t)> modelGetter, std::string methodName,
			size_t countIteration, size_t startDim, size_t finishDim, size_t stepDim);

	void RealAndModelTimeRelation2D(std::function<IterativeModel*(size_t, size_t)> modelGetter, std::string methodName,
		size_t maxTimeSec, size_t stepTimeSec, size_t beginDim, size_t endDim, size_t stepDim);
}