#include "Heat2DModel.h"

namespace Heat2D {
	Heat2DModel::Heat2DModel(HostData2D<float> t, float h, float a, float tau)
			: IterativeModel(tau), _t(t), _h(h), _a(a), isSweepCpuInit(false), isSweepGpuInit(false) {
		fnCPU = Heat2DFunctor<HostData2D<float>>(a, h, tau, t);

		AddCalculationMethod("cpu_gaussseidel", std::bind(std::mem_fun(&Heat2DModel::CalculationMethod_CPU_GaussSeidel), this));
		AddCalculationMethod("cpu_sweep", std::bind(std::mem_fun(&Heat2DModel::CalculationMethod_CPU_Sweep), this));
		AddCalculationMethod("gpu_gaussseidel_direct", std::bind(std::mem_fun(&Heat2DModel::CalculationMethod_GPU_GaussSeidel_Direct), this));
		AddCalculationMethod("gpu_gaussseidel_chess", std::bind(std::mem_fun(&Heat2DModel::CalculationMethod_GPU_GaussSeidel_Chess), this));
		AddCalculationMethod("gpu_gaussseidel_withoutconflicts", std::bind(std::mem_fun(&Heat2DModel::CalculationMethod_GPU_GaussSeidel_WithoutConflicts), this));
		AddCalculationMethod("gpu_gaussseidel_direct_overlay", std::bind(std::mem_fun(&Heat2DModel::CalculationMethod_GPU_GaussSeidel_DirectOverlay), this));
		AddCalculationMethod("gpu_sweep_linedevide", std::bind(std::mem_fun(&Heat2DModel::CalculationMethod_GPU_Sweep_LineDevide), this));
		AddCalculationMethod("gpu_sweep_blockdevide", std::bind(std::mem_fun(&Heat2DModel::CalculationMethod_GPU_Sweep_BlockDevide), this));
	}

	std::string Heat2DModel::PrintData() const {
		return "";//HostDataPrinter::Print<float>(t);
	}

	void Heat2DModel::SynchronizeWithGpu() {
		if (isGpuOn())
			_t_dev.PutTo(_t);
	}

	void Heat2DModel::PrepareDataForGpu(const Sonsode::GpuDevice &gpu, size_t orderNumber) throw(std::string) {
		_t_dev = DeviceData2D<float>(gpu, _t);
		fnGPU = Heat2DFunctor<DeviceData2D<float>>(a(), h(), tau(), _t_dev);
	}

	void Heat2DModel::FreeDataForGpus() throw(std::string) {
		_t_dev.Erase();
	}

	#pragma region Calculation methods

	void Heat2DModel::CalculationMethod_CPU_GaussSeidel() {
		GpuOff();
		ExplicitGaussSeidel_2D_CPU(fnCPU);
	}

	void Heat2DModel::CalculationMethod_CPU_Sweep() {
		GpuOff();
		InitSweep(false);
		ImplicitSweep_2D_CPU(sf_h, fnCPU);
	}

	void Heat2DModel::CalculationMethod_GPU_GaussSeidel_Direct() {
		GpuOn();
		ExplicitGaussSeidel_2D_GPU_direct(fnGPU);
	}

	void Heat2DModel::CalculationMethod_GPU_GaussSeidel_Chess() {
		GpuOn();
		ExplicitGaussSeidel_2D_GPU_chess(fnGPU);
	}

	void Heat2DModel::CalculationMethod_GPU_GaussSeidel_WithoutConflicts() {
		GpuOn();
		ExplicitGaussSeidel_2D_GPU_outconf(fnGPU);
	}

	void Heat2DModel::CalculationMethod_GPU_GaussSeidel_DirectOverlay() {
		GpuOn();
		ExplicitGaussSeidel_2D_GPU_directOverlay(fnGPU);
	}

	void Heat2DModel::CalculationMethod_GPU_Sweep_LineDevide() {
		GpuOn();
		InitSweep(true);
		ImplicitSweep_2D_GPU_lineDivide(sf_d, fnGPU);
	}

	void Heat2DModel::CalculationMethod_GPU_Sweep_BlockDevide() {
		GpuOn();
		InitSweep(true);
		ImplicitSweep_2D_GPU_blockDivide(sf_d, fnGPU);
	}

	#pragma endregion

	void Heat2DModel::InitSweep(bool useGpu) throw (std::string) {
		if (!useGpu && !isSweepCpuInit) {
			//Init CPU sweep
			sf_h = HostData2D<SweepFactors<float>>(_t.dimX(), _t.dimY());
			isSweepCpuInit = true;
		}

		if (useGpu && !isSweepGpuInit && isGpuOn()) {
			//Init GPU sweep
			sf_d = DeviceData2D<SweepFactors<float>>(_t_dev.gpu(), _t_dev.dimX(), _t_dev.dimY());
			isSweepGpuInit = true;
		}
	}

	void Heat2DModel::DeinitSweep() throw (std::string) {
		if (isSweepCpuInit) {
			sf_h.Erase();
			isSweepCpuInit = false;
		}

		if (isSweepGpuInit) {
			sf_d.Erase();
			isSweepGpuInit = false;
		}
	}
}