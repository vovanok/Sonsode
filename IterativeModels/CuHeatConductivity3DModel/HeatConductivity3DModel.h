#pragma once

#include <functional>
#include "IterativeModel.h"
#include "HeatConductivity3D_Functors.cu"
#include "SonsodeFunctionsLib.cu"
#include "HostDataPrinter.hpp"
#include "SweepFactors.cu"
#include "HostData.hpp"
#include "DeviceData.cu"

using Sonsode::HostData3D;
using Sonsode::DeviceData3D;
using Sonsode::SweepFactors;
using Sonsode::FunctionsLib::ExplicitGaussSeidel_3D_CPU;
using Sonsode::FunctionsLib::ExplicitGaussSeidel_3D_GPU_direct;
using Sonsode::FunctionsLib::ImplicitSweep_3D_CPU;
using Sonsode::FunctionsLib::ImplicitSweep_3D_GPU_lineDivide;

class HeatConductivity3DModel : public IterativeModel {
protected:
	HostData3D<float> _t;
	DeviceData3D<float> _t_dev;
	float _h;
	float _a;

	bool isSweepCpuInit;
	bool isSweepGpuInit;
	HostData3D<SweepFactors<float>> sf_h;
	DeviceData3D<SweepFactors<float>> sf_d;

	HeatConductivity3DFunctor<HostData3D<float>> fnCPU;
	HeatConductivity3DFunctor<DeviceData3D<float>> fnGPU;

	//BaseModel
	virtual void PrepareDataForGpu(const Sonsode::GpuDevice &gpu, size_t orderNumber);
	virtual void FreeDataForGpus();

	void CalculationMethod_CPU_GaussSeidel();
	void CalculationMethod_CPU_Sweep();
	void CalculationMethod_GPU_GaussSeidel_Direct();
	void CalculationMethod_GPU_Sweep_LineDevide();

	void InitSweep(bool useGpu) throw (std::string);
	void DeinitSweep() throw (std::string);
public:
	HeatConductivity3DModel(HostData3D<float> t, float h, float a, float tau);
	~HeatConductivity3DModel() {
		GpuOff();
		DeinitSweep();
		_t.Erase();
	}

	virtual std::string PrintData() const;
	virtual void SynchronizeWithGpu();

	float h() const { return _h; }
	float a() const { return _a; }
	HostData3D<float> t() { return _t; }
};