#pragma once

#include <string>
#include "IterativeModel.h"
#include "Vectors.hpp"
#include "HostData.hpp"
#include "DeviceData.cu"
#include "OilSpillFunctors.cu"
#include "SonsodeFunctionsLib.cu"
#include "HostDataPrinter.hpp"
#include "SweepFactors.cu"
#include "OilSpillPodTypes.cu"

namespace OilSpill {
	using Sonsode::GpuDevice;
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
	using namespace Functors;

	class OilSpillModel : public IterativeModel {
	public:
		virtual std::string PrintData() const;
		virtual void SynchronizeWithGpu();

		OilSpillModel(OilSpillConsts consts, OilSpillDataH data);
		virtual ~OilSpillModel();

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
		OilSpillDataH _data;
		OilSpillDataD _data_dev;

		OilSpillConsts _consts;

		HostData2D<SweepFactors<float>> sf_h;
		DeviceData2D<SweepFactors<float>> sf_d;

		WaterUx<OilSpillDataH> waterUxCPU;
		WaterUy<OilSpillDataH> waterUyCPU;
		Impurity<OilSpillDataH> impurityCPU;
		Press<OilSpillDataH> pressCPU;
		WaterUxS<OilSpillDataH> waterUxSCPU;
		WaterUyS<OilSpillDataH> waterUySCPU;
		WaterUx_Complete<OilSpillDataH> waterUx_CompleteCPU;
		WaterUy_Complete<OilSpillDataH> waterUy_CompleteCPU;
		OilUx<OilSpillDataH> oilUxCPU;
		OilUy<OilSpillDataH> oilUyCPU;
		PressS<OilSpillDataH> pressSCPU;
		W<OilSpillDataH> wCPU;
		ImpurityIstok<OilSpillDataH> impurityIstokCPU;
		ImpurityS<OilSpillDataH> impuritySCPU;
		IslandResolver<OilSpillDataH> islandResolverCPU;

		WaterUx<OilSpillDataD> waterUxGPU;
		WaterUy<OilSpillDataD> waterUyGPU;
		Impurity<OilSpillDataD> impurityGPU;
		Press<OilSpillDataD> pressGPU;
		WaterUxS<OilSpillDataD> waterUxSGPU;
		WaterUyS<OilSpillDataD> waterUySGPU;
		WaterUx_Complete<OilSpillDataD> waterUx_CompleteGPU;
		WaterUy_Complete<OilSpillDataD> waterUy_CompleteGPU;
		OilUx<OilSpillDataD> oilUxGPU;
		OilUy<OilSpillDataD> oilUyGPU;
		PressS<OilSpillDataD> pressSGPU;
		W<OilSpillDataD> wGPU;
		ImpurityIstok<OilSpillDataD> impurityIstokGPU;
		ImpurityS<OilSpillDataD> impuritySGPU;
		IslandResolver<OilSpillDataD> islandResolverGPU;

	protected:
		virtual void PrepareDataForGpu(const GpuDevice &gpuDevice, size_t orderNumber) throw (std::string);
		virtual void FreeDataForGpus() throw (std::string);

		void CalculationMethod_CPU();
		void CalculationMethod_GPU();
	};
}