#pragma once

#include <string>
#include "IterativeModel.h"
#include "HostData.hpp"
#include "DeviceData.cu"
#include "HostDataPrinter.hpp"
#include "AirFlowDistributionFunctors.cu"
#include "AirFlowDistributionModelPodTypes.cu"
#include "SweepFactors.cu"
#include "SonsodeFunctionsLib.cu"

using Sonsode::HostData3D;
using Sonsode::DeviceData3D;
using Sonsode::SweepFactors;
using Sonsode::GpuDevice;
using Sonsode::GpuDeviceFactory;
using Sonsode::FunctionsLib::ImplicitSweep_3D_CPU;
using Sonsode::FunctionsLib::ImplicitSweep_3D_GPU_lineDivide;
using Sonsode::FunctionsLib::Boundary_3D_CPU;
using Sonsode::FunctionsLib::Boundary_3D_GPU;

class AirFlowDistributionModel : public IterativeModel {
public:
	AirFlowDistributionModel(AirFlowConsts consts, AirFlowDataH data);
	~AirFlowDistributionModel();
	virtual std::string PrintData() const;
	virtual void SynchronizeWithGpu();

protected:
	AirFlowConsts _consts;
	AirFlowDataH _data;
	AirFlowDataD _data_dev;

	HostData3D<SweepFactors<float>> sf_h;
	DeviceData3D<SweepFactors<float>> sf_d;
	
	AirFlowDistributionFunctor::Ro<AirFlowDataH> roCPU;
	AirFlowDistributionFunctor::Ux<AirFlowDataH> uxCPU;
	AirFlowDistributionFunctor::Uy<AirFlowDataH> uyCPU;
	AirFlowDistributionFunctor::Uz<AirFlowDataH> uzCPU;

	AirFlowDistributionFunctor::Ro<AirFlowDataD> roGPU;
	AirFlowDistributionFunctor::Ux<AirFlowDataD> uxGPU;
	AirFlowDistributionFunctor::Uy<AirFlowDataD> uyGPU;
	AirFlowDistributionFunctor::Uz<AirFlowDataD> uzGPU;

	//void ApplySpecialArea(SpecialAirArea specialArea) {
	//	for (size_t x = 0; x < _data.dimX(); x++)
	//		for (size_t y = 0; y < _data.dimY(); y++)
	//			for (size_t z = 0; z < _data.dimZ(); z++)
	//				if (specialArea.IsInside(x, y, z))
	//					_data(x, y, z) = specialArea.SpecialValue;
	//}

protected:
	virtual void PrepareDataForGpu(const GpuDevice &gpuDevice, size_t orderNumber);
	virtual void FreeDataForGpus();

	void CalculationMethod_CPU();
	void CalculationMethod_GPU();
};