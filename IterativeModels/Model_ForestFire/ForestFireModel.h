#pragma once

#include <functional>
#include "IterativeModel.h"
#include "SonsodeFunctionsLib.cu"
#include "HostDataPrinter.hpp"
#include "SweepFactors.cu"
#include "HostData.hpp"
#include "DeviceData.cu"
#include "Vectors.hpp"
#include "ForestFirePodTypes.cu"
#include "ForestFireFunctors.cu"

namespace ForestFire {
	using Sonsode::GpuDevice;
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
	using namespace Functors;

	class ForestFireModel : public IterativeModel {
	public:
		virtual std::string PrintData() const;
		virtual void SynchronizeWithGpu();

		ForestFireModel(ForestFireConsts consts, ForestFireDataH data);
		virtual ~ForestFireModel();

	protected:
		ForestFireDataH _data;
		ForestFireDataD _data_dev;

		ForestFireConsts _consts;

		HostData2D<SweepFactors<float>> sf_h;
		DeviceData2D<SweepFactors<float>> sf_d;
	
		T4<ForestFireDataH> t4CPU;
		Gorenie<ForestFireDataH> gorenieCPU;
		Temperature<ForestFireDataH> temperatureCPU;

		T4<ForestFireDataD> t4GPU;
		Gorenie<ForestFireDataD> gorenieGPU;
		Temperature<ForestFireDataD> temperatureGPU;

		void CalculationMethod_CPU();
		void CalculationMethod_GPU();

		virtual void PrepareDataForGpu(const GpuDevice &gpu, size_t orderNumber);
		virtual void FreeDataForGpus();
	};
}