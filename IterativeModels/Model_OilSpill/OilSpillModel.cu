#include "OilSpillModel.h"

namespace OilSpill {
	OilSpillModel::OilSpillModel(OilSpillConsts consts, OilSpillDataH data)
			: IterativeModel(consts.Tau), _data(data), _consts(consts) {
		data.FillS(0.0f);
		sf_h = HostData2D<SweepFactors<float>>(data.dimX(), data.dimY());

		waterUxCPU = WaterUx<OilSpillDataH>(consts, data);
		waterUyCPU = WaterUy<OilSpillDataH>(consts, data);
		impurityCPU = Impurity<OilSpillDataH>(consts, data);
		pressCPU = Press<OilSpillDataH>(consts, data);
		waterUxSCPU = WaterUxS<OilSpillDataH>(consts, data);
		waterUySCPU = WaterUyS<OilSpillDataH>(consts, data);
		waterUx_CompleteCPU = WaterUx_Complete<OilSpillDataH>(consts, data);
		waterUy_CompleteCPU = WaterUy_Complete<OilSpillDataH>(consts, data);
		oilUxCPU = OilUx<OilSpillDataH>(consts, data);
		oilUyCPU = OilUy<OilSpillDataH>(consts, data);
		pressSCPU = PressS<OilSpillDataH>(consts, data);
		wCPU = W<OilSpillDataH>(consts, data);
		impurityIstokCPU = ImpurityIstok<OilSpillDataH>(consts, data);
		impuritySCPU = ImpurityS<OilSpillDataH>(consts, data);
		islandResolverCPU = IslandResolver<OilSpillDataH>(consts, data);

		AddCalculationMethod("cpu", std::bind(std::mem_fun(&OilSpillModel::CalculationMethod_CPU), this));
		AddCalculationMethod("gpu", std::bind(std::mem_fun(&OilSpillModel::CalculationMethod_GPU), this));
	}

	OilSpillModel::~OilSpillModel() {
		GpuOff();

		_data.Erase();
		sf_h.Erase();
	}

	std::string OilSpillModel::PrintData() const {
		return "";
	}

	void OilSpillModel::SynchronizeWithGpu() {
		if (isGpuOn())
			_data_dev.PutTo(_data);
	}

	void OilSpillModel::PrepareDataForGpu(const GpuDevice &gpuDevice, size_t orderNumber) {
		sf_d = DeviceData2D<SweepFactors<float>>(gpuDevice, sf_h);
		_data_dev = OilSpillDataD(gpuDevice, _data);

		waterUxGPU = WaterUx<OilSpillDataD>(_consts, _data_dev);
		waterUyGPU = WaterUy<OilSpillDataD>(_consts, _data_dev);
		impurityGPU = Impurity<OilSpillDataD>(_consts, _data_dev);
		pressGPU = Press<OilSpillDataD>(_consts, _data_dev);
		waterUxSGPU = WaterUxS<OilSpillDataD>(_consts, _data_dev);
		waterUySGPU = WaterUyS<OilSpillDataD>(_consts, _data_dev);
		waterUx_CompleteGPU = WaterUx_Complete<OilSpillDataD>(_consts, _data_dev);
		waterUy_CompleteGPU = WaterUy_Complete<OilSpillDataD>(_consts, _data_dev);
		oilUxGPU = OilUx<OilSpillDataD>(_consts, _data_dev);
		oilUyGPU = OilUy<OilSpillDataD>(_consts, _data_dev);
		pressSGPU = PressS<OilSpillDataD>(_consts, _data_dev);
		wGPU = W<OilSpillDataD>(_consts, _data_dev);
		impurityIstokGPU = ImpurityIstok<OilSpillDataD>(_consts, _data_dev);
		impuritySGPU = ImpurityS<OilSpillDataD>(_consts, _data_dev);
		islandResolverGPU = IslandResolver<OilSpillDataD>(_consts, _data_dev);
	}

	void OilSpillModel::FreeDataForGpus() {
		_data_dev.Erase();
		sf_d.Erase();
	}

	void OilSpillModel::CalculationMethod_CPU() {
		GpuOff();
	
		ImplicitSweep_2D_CPU(sf_h, waterUxCPU);
		ImplicitSweep_2D_CPU(sf_h, waterUyCPU);

		ExplicitGaussSeidel_2D_CPU(pressSCPU);

		Boundary_2D_CPU(pressSCPU);
		Boundary_2D_CPU(impuritySCPU);

		ImplicitSweep_2D_CPU(sf_h, pressCPU);

		Boundary_2D_CPU(pressCPU);

		ExplicitGaussSeidel_2D_CPU(waterUxSCPU);
		ExplicitGaussSeidel_2D_CPU(waterUySCPU);

		ExplicitGaussSeidel_2D_CPU(waterUx_CompleteCPU);
		ExplicitGaussSeidel_2D_CPU(waterUy_CompleteCPU);

		ExplicitGaussSeidel_2D_CPU(oilUxCPU);
		ExplicitGaussSeidel_2D_CPU(oilUyCPU);

		ExplicitGaussSeidel_2D_CPU(wCPU);
		ImplicitSweep_2D_CPU(sf_h, impurityCPU);

		ExplicitGaussSeidel_2D_CPU(impurityIstokCPU);

		FullSearch_2D_CPU(islandResolverCPU);
	}

	void OilSpillModel::CalculationMethod_GPU() {
		GpuOn();

		ImplicitSweep_2D_GPU_lineDivide(sf_d, waterUxGPU);
		ImplicitSweep_2D_GPU_lineDivide(sf_d, waterUyGPU);

		ExplicitGaussSeidel_2D_GPU_direct(pressSGPU);

		Boundary_2D_GPU(pressSGPU);
		Boundary_2D_GPU(impuritySGPU);

		ImplicitSweep_2D_GPU_lineDivide(sf_d, pressGPU);

		Boundary_2D_GPU(pressGPU);

		ExplicitGaussSeidel_2D_GPU_direct(waterUxSGPU);
		ExplicitGaussSeidel_2D_GPU_direct(waterUySGPU);

		ExplicitGaussSeidel_2D_GPU_direct(waterUx_CompleteGPU);
		ExplicitGaussSeidel_2D_GPU_direct(waterUy_CompleteGPU);

		ExplicitGaussSeidel_2D_GPU_direct(oilUxGPU);
		ExplicitGaussSeidel_2D_GPU_direct(oilUyGPU);

		ExplicitGaussSeidel_2D_GPU_direct(wGPU);
		ImplicitSweep_2D_GPU_lineDivide(sf_d, impurityGPU);

		ExplicitGaussSeidel_2D_GPU_direct(impurityIstokGPU);

		FullSearch_2D_GPU(islandResolverGPU);
	}
}