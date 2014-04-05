#pragma once

#include <functional>
#include "IterativeModel.h"
#include "SonsodeFunctionsLib.cu"
#include "HostDataPrinter.hpp"
#include "SweepFactors.cu"
#include "HostData.hpp"
#include "DeviceData.cu"
#include "Vectors.hpp"
#include "FireSpreadSimpleModelPodTypes.cu"
#include "FireSpreadSimpleModelFunctors.cu"

using Sonsode::HostData1D;
using Sonsode::HostData2D;
using Sonsode::DeviceData1D;
using Sonsode::DeviceData2D;
using Sonsode::SweepFactors;
using Sonsode::FunctionsLib::ExplicitGaussSeidel_2D_CPU;
using Sonsode::FunctionsLib::ExplicitGaussSeidel_2D_GPU_direct;
using Sonsode::FunctionsLib::ImplicitSweep_2D_CPU;
using Sonsode::FunctionsLib::ImplicitSweep_2D_GPU_lineDivide;
using Sonsode::FunctionsLib::FullSearch_2D_CPU;
using Sonsode::FunctionsLib::FullSearch_2D_GPU;

class FireSpreadSimpleModel : public IterativeModel {
public:
	virtual std::string PrintData() const;
	virtual void SynchronizeWithGpu();

	FireSpreadSimpleModel(FireSpreadConsts consts, FireSpreadDataH data);
	virtual ~FireSpreadSimpleModel();

protected:
	FireSpreadDataH _data;
	FireSpreadDataD _data_dev;

	FireSpreadConsts _consts;

	HostData2D<SweepFactors<float>> sf_h;
	DeviceData2D<SweepFactors<float>> sf_d;
	
	FireSpreadFunctor::T4<FireSpreadDataH> t4CPU;
	FireSpreadFunctor::Gorenie<FireSpreadDataH> gorenieCPU;
	FireSpreadFunctor::Temperature<FireSpreadDataH> temperatureCPU;

	FireSpreadFunctor::T4<FireSpreadDataD> t4GPU;
	FireSpreadFunctor::Gorenie<FireSpreadDataD> gorenieGPU;
	FireSpreadFunctor::Temperature<FireSpreadDataD> temperatureGPU;

	void CalculationMethod_CPU();
	void CalculationMethod_GPU();

	virtual void PrepareDataForGpu(const Sonsode::GpuDevice &gpu, size_t orderNumber);
	virtual void FreeDataForGpus();
};