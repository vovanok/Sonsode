#pragma once

#include <string>
#include "IterativeModel.h"
#include "Vectors.hpp"
#include "HostData.hpp"
#include "DeviceData.cu"
#include "OilSpillageImprovedModelFunctors.cu"
#include "SonsodeFunctionsLib.cu"
#include "HostDataPrinter.hpp"
#include "SweepFactors.cu"
#include "OilSpillageImprovedModelPodTypes.cu"

using Sonsode::HostData2D;
using Sonsode::SweepFactors;
using Sonsode::FunctionsLib::ImplicitSweep_2D_CPU;
using Sonsode::FunctionsLib::ImplicitSweep_2D_GPU_lineDivide;
using Sonsode::FunctionsLib::ExplicitGaussSeidel_2D_CPU;
using Sonsode::FunctionsLib::ExplicitGaussSeidel_2D_GPU_direct;
using Sonsode::FunctionsLib::Boundary_2D_CPU;
using Sonsode::FunctionsLib::Boundary_2D_GPU;
using Sonsode::FunctionsLib::FullSearch_2D_CPU;
using Sonsode::FunctionsLib::FullSearch_2D_GPU;

class OilSpillageImprovedModel : public IterativeModel {
public:
	virtual std::string PrintData() const;
	virtual void SynchronizeWithGpu();

	OilSpillageImprovedModel(OilSpillageConsts consts, OilSpillageDataH data);
	virtual ~OilSpillageImprovedModel();

	float& w(size_t x, size_t y) { return _data.w(x, y); }
	float& waterUx(size_t x, size_t y) { return _data.waterUx(x, y); }
	float& waterUy(size_t x, size_t y) { return _data.waterUy(x, y); }
	float& oilUx(size_t x, size_t y) { return _data.oilUx(x, y); }
	float& oilUy(size_t x, size_t y) { return _data.oilUy(x, y); }
	float& deep(size_t x, size_t y) { return _data.deep(x, y); }
	float& impurity(size_t x, size_t y) { return _data.impurity(x, y); }
	float& press(size_t x, size_t y) { return _data.press(x, y); }

	size_t dimX() { return _data.dimX(); }
	size_t dimY() { return _data.dimX(); }

private:
	OilSpillageDataH _data;
	OilSpillageDataD _data_dev;

	OilSpillageConsts _consts;

	HostData2D<SweepFactors<float>> sf_h;
	DeviceData2D<SweepFactors<float>> sf_d;

	OilSpillageImprovedFunctor::WaterUx<OilSpillageDataH> waterUxCPU;
	OilSpillageImprovedFunctor::WaterUy<OilSpillageDataH> waterUyCPU;
	OilSpillageImprovedFunctor::Impurity<OilSpillageDataH> impurityCPU;
	OilSpillageImprovedFunctor::Press<OilSpillageDataH> pressCPU;
	OilSpillageImprovedFunctor::WaterUxS<OilSpillageDataH> waterUxSCPU;
	OilSpillageImprovedFunctor::WaterUyS<OilSpillageDataH> waterUySCPU;
	OilSpillageImprovedFunctor::WaterUx_Complete<OilSpillageDataH> waterUx_CompleteCPU;
	OilSpillageImprovedFunctor::WaterUy_Complete<OilSpillageDataH> waterUy_CompleteCPU;
	OilSpillageImprovedFunctor::OilUx<OilSpillageDataH> oilUxCPU;
	OilSpillageImprovedFunctor::OilUy<OilSpillageDataH> oilUyCPU;
	OilSpillageImprovedFunctor::PressS<OilSpillageDataH> pressSCPU;
	OilSpillageImprovedFunctor::W<OilSpillageDataH> wCPU;
	OilSpillageImprovedFunctor::ImpurityIstok<OilSpillageDataH> impurityIstokCPU;
	OilSpillageImprovedFunctor::ImpurityS<OilSpillageDataH> impuritySCPU;
	OilSpillageImprovedFunctor::IslandResolver<OilSpillageDataH> islandResolverCPU;

	OilSpillageImprovedFunctor::WaterUx<OilSpillageDataD> waterUxGPU;
	OilSpillageImprovedFunctor::WaterUy<OilSpillageDataD> waterUyGPU;
	OilSpillageImprovedFunctor::Impurity<OilSpillageDataD> impurityGPU;
	OilSpillageImprovedFunctor::Press<OilSpillageDataD> pressGPU;
	OilSpillageImprovedFunctor::WaterUxS<OilSpillageDataD> waterUxSGPU;
	OilSpillageImprovedFunctor::WaterUyS<OilSpillageDataD> waterUySGPU;
	OilSpillageImprovedFunctor::WaterUx_Complete<OilSpillageDataD> waterUx_CompleteGPU;
	OilSpillageImprovedFunctor::WaterUy_Complete<OilSpillageDataD> waterUy_CompleteGPU;
	OilSpillageImprovedFunctor::OilUx<OilSpillageDataD> oilUxGPU;
	OilSpillageImprovedFunctor::OilUy<OilSpillageDataD> oilUyGPU;
	OilSpillageImprovedFunctor::PressS<OilSpillageDataD> pressSGPU;
	OilSpillageImprovedFunctor::W<OilSpillageDataD> wGPU;
	OilSpillageImprovedFunctor::ImpurityIstok<OilSpillageDataD> impurityIstokGPU;
	OilSpillageImprovedFunctor::ImpurityS<OilSpillageDataD> impuritySGPU;
	OilSpillageImprovedFunctor::IslandResolver<OilSpillageDataD> islandResolverGPU;

	void ResolveBoundaryConditions();
protected:
	virtual void PrepareDataForGpu(const Sonsode::GpuDevice &gpuDevice, size_t orderNumber) throw (std::string);
	virtual void FreeDataForGpus() throw (std::string);

	void CalculationMethod_CPU();
	void CalculationMethod_GPU();
};