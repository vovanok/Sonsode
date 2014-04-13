#pragma once

#include <string>
#include "IterativeModel.h"
#include "HostData.hpp"
#include "DeviceData.cu"
#include "HostDataPrinter.hpp"
#include "AirFlowFunctors.cu"
#include "AirFlowPodTypes.cu"
#include "SweepFactors.cu"
#include "SonsodeFunctionsLib.cu"

namespace AirFlow {
	using Sonsode::HostData3D;
	using Sonsode::DeviceData3D;
	using Sonsode::SweepFactors;
	using Sonsode::GpuDevice;
	using Sonsode::GpuDeviceFactory;
	using Sonsode::FunctionsLib::ImplicitSweep_3D_CPU;
	using Sonsode::FunctionsLib::ImplicitSweep_3D_GPU_lineDivide;
	using Sonsode::FunctionsLib::Boundary_3D_CPU;
	using Sonsode::FunctionsLib::Boundary_3D_GPU;
	using namespace Functors;

	class AirFlowModel : public IterativeModel {
	public:
		AirFlowModel(AirFlowConsts consts, AirFlowDataH data);
		~AirFlowModel();
		virtual std::string PrintData() const;
		virtual void SynchronizeWithGpu();

	protected:
		AirFlowConsts _consts;
		AirFlowDataH _data;
		AirFlowDataD _data_dev;

		HostData3D<SweepFactors<float>> sf_h;
		DeviceData3D<SweepFactors<float>> sf_d;
	
		Ro<AirFlowDataH> roCPU;
		Ux<AirFlowDataH> uxCPU;
		Uy<AirFlowDataH> uyCPU;
		Uz<AirFlowDataH> uzCPU;

		Ro<AirFlowDataD> roGPU;
		Ux<AirFlowDataD> uxGPU;
		Uy<AirFlowDataD> uyGPU;
		Uz<AirFlowDataD> uzGPU;

	protected:
		virtual void PrepareDataForGpu(const GpuDevice &gpuDevice, size_t orderNumber);
		virtual void FreeDataForGpus();

		void CalculationMethod_CPU();
		void CalculationMethod_GPU();
	};
}