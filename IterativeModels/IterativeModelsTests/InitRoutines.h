#pragma once

#include <iostream>
#include "IterativeModel.h"
#include "FireSpreadSimpleModelTest.h"
#include "HeatConductivity3DModelTest.h"
#include "HeatConductivity2DModelTest.h"
//#include "HeatConductivitySweepModelTest.h"
//#include "ParticlesModelTest.h"
//#include "OilSpillageModelTest.h"
#include "AirFlowDistributionModelTest.h"
//#include "FireSpread3DModelTest.h"
#include "OilSpillageImprovedModelTest.h"
//#include "Model_ExperimentalTest.h"

namespace InitRoutines {
	//ParticlesModelTest *GetInitedParticlesModel();
	HeatConductivity2DModelTest* GetInitedHeatConductivityModel();
	HeatConductivity2DModelTest* GetInitedHeatConductivityModel(size_t dimX, size_t dimY);
	HeatConductivity3DModelTest* GetInitedHeatConductivity3DModel();
	HeatConductivity3DModelTest* GetInitedHeatConductivity3DModel(size_t dimX, size_t dimY, size_t dimZ);
	FireSpreadSimpleModelTest* GetInitedFireSpreadSimpleModel();
	FireSpreadSimpleModelTest* GetInitedFireSpreadSimpleModel(size_t dimX, size_t dimY);
	FireSpreadSimpleModelTest *GetInitedFireSpreadModelTest();
	AirFlowDistributionModelTest* GetInitedAirFlowDistributionModel();
	AirFlowDistributionModelTest* GetInitedAirFlowDistributionModel(size_t dimX, size_t dimY, size_t dimZ);
	//OilSpillageModelTest* GetInitedOilSpillageModel();
	//FireSpread3DModelTest* GetInitedFireSpread3DModel();
	OilSpillageImprovedModelTest* GetInitedOilSpillageImprovedModel();
	OilSpillageImprovedModelTest* GetInitedOilSpillageImprovedModel(size_t dimX, size_t dimY);

	IterativeModel *GetModelByUserChange();
}