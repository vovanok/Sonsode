#include "HeatConductivity3DModel.h"

HeatConductivity3DModel::HeatConductivity3DModel(HostData3D<float> t, float h, float a, float tau)
		: IterativeModel(tau), _t(t), _h(h), _a(a), isSweepCpuInit(false), isSweepGpuInit(false) {
	fnCPU = HeatConductivity3DFunctor<HostData3D<float>>(a, h, tau, t);
	
	AddCalculationMethod("cpu_gaussseidel", std::bind(std::mem_fun(&HeatConductivity3DModel::CalculationMethod_CPU_GaussSeidel), this));
	AddCalculationMethod("cpu_sweep", std::bind(std::mem_fun(&HeatConductivity3DModel::CalculationMethod_CPU_Sweep), this));
	AddCalculationMethod("gpu_gaussseidel_direct", std::bind(std::mem_fun(&HeatConductivity3DModel::CalculationMethod_GPU_GaussSeidel_Direct), this));
	AddCalculationMethod("gpu_sweep_linedevide", std::bind(std::mem_fun(&HeatConductivity3DModel::CalculationMethod_GPU_Sweep_LineDevide), this));
}

std::string HeatConductivity3DModel::PrintData() const {
	return "";
}

void HeatConductivity3DModel::SynchronizeWithGpu() {
	if (isGpuOn())
		_t_dev.PutTo(_t);
}

void HeatConductivity3DModel::PrepareDataForGpu(const Sonsode::GpuDevice &gpu, size_t orderNumber) {
	_t_dev = DeviceData3D<float>(gpu, _t);
	fnGPU = HeatConductivity3DFunctor<DeviceData3D<float>>(a(), h(), tau(), _t_dev);
}

void HeatConductivity3DModel::FreeDataForGpus() {
	_t_dev.Erase();
}

#pragma region Calculation methods

void HeatConductivity3DModel::CalculationMethod_CPU_GaussSeidel() {
	GpuOff();
	ExplicitGaussSeidel_3D_CPU(fnCPU);
}

void HeatConductivity3DModel::CalculationMethod_CPU_Sweep() {
	GpuOff();
	InitSweep(false);
	ImplicitSweep_3D_CPU(sf_h, fnCPU);
}

void HeatConductivity3DModel::CalculationMethod_GPU_GaussSeidel_Direct() {
	GpuOn();
	ExplicitGaussSeidel_3D_GPU_direct(fnGPU);
}

void HeatConductivity3DModel::CalculationMethod_GPU_Sweep_LineDevide() {
	GpuOn();
	InitSweep(true);
	ImplicitSweep_3D_GPU_lineDivide(sf_d, fnGPU);
}

#pragma endregion

void HeatConductivity3DModel::InitSweep(bool useGpu) throw (std::string) {
	if (!useGpu && !isSweepCpuInit) {
		//Init CPU sweep
		sf_h = HostData3D<SweepFactors<float>>(_t.dimX(), _t.dimY(), _t.dimZ());
		isSweepCpuInit = true;
	}

	if (useGpu && !isSweepGpuInit && isGpuOn()) {
		//Init GPU sweep
		sf_d = DeviceData3D<SweepFactors<float>>(_t_dev.gpu(), _t_dev.dimX(), _t_dev.dimY(), _t_dev.dimZ());
		isSweepGpuInit = true;
	}
}

void HeatConductivity3DModel::DeinitSweep() throw (std::string) {
	if (isSweepCpuInit) {
		sf_h.Erase();
		isSweepCpuInit = false;
	}

	if (isSweepGpuInit) {
		sf_d.Erase();
		isSweepGpuInit = false;
	}
}