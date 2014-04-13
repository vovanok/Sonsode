#include "AirFlowModel.h"

namespace AirFlow {
	AirFlowModel::AirFlowModel(AirFlowConsts consts, AirFlowDataH data)
			: IterativeModel(consts.Tau), _consts(consts), _data(data) {

		sf_h = HostData3D<SweepFactors<float>>(data.dimX(), data.dimY(), data.dimZ());

		roCPU = Ro<AirFlowDataH>(_consts, _data);
		uxCPU = Ux<AirFlowDataH>(_consts, _data);
		uyCPU = Uy<AirFlowDataH>(_consts, _data);
		uzCPU = Uz<AirFlowDataH>(_consts, _data);

		AddCalculationMethod("cpu", std::bind(std::mem_fun(&AirFlowModel::CalculationMethod_CPU), this));
		AddCalculationMethod("gpu", std::bind(std::mem_fun(&AirFlowModel::CalculationMethod_GPU), this));
	}

	AirFlowModel::~AirFlowModel() {
		GpuOff();

		_data.Erase();
		sf_h.Erase();
	}

	std::string AirFlowModel::PrintData() const {
		return "";//HostDataPrinter::Print<AirFlowPoint>(md->Data, AirFlowDistributionModelData::PrintAirFlowPoint);
	}

	void AirFlowModel::SynchronizeWithGpu() {
		if (isGpuOn())
			_data_dev.PutTo(_data);
	}

	void AirFlowModel::PrepareDataForGpu(const GpuDevice& gpuDevice, size_t orderNumber) {
		sf_d = DeviceData3D<SweepFactors<float>>(gpuDevice, sf_h);
		_data_dev = AirFlowDataD(gpuDevice, _data);

		roGPU = Ro<AirFlowDataD>(_consts, _data_dev);
		uxGPU = Ux<AirFlowDataD>(_consts, _data_dev);
		uyGPU = Uy<AirFlowDataD>(_consts, _data_dev);
		uzGPU = Uz<AirFlowDataD>(_consts, _data_dev);
	}

	void AirFlowModel::FreeDataForGpus() {
		_data_dev.Erase();
		sf_d.Erase();
	}

	void AirFlowModel::CalculationMethod_CPU() {
		ImplicitSweep_3D_CPU(sf_h, roCPU);
		ImplicitSweep_3D_CPU(sf_h, uxCPU);
		ImplicitSweep_3D_CPU(sf_h, uyCPU);
		ImplicitSweep_3D_CPU(sf_h, uzCPU);

		Boundary_3D_CPU(roCPU);
		Boundary_3D_CPU(uxCPU);
		Boundary_3D_CPU(uyCPU);
		Boundary_3D_CPU(uzCPU);
	}

	void AirFlowModel::CalculationMethod_GPU() {
		GpuOn();

		ImplicitSweep_3D_GPU_lineDivide(sf_d, roGPU);
		ImplicitSweep_3D_GPU_lineDivide(sf_d, uxGPU);
		ImplicitSweep_3D_GPU_lineDivide(sf_d, uyGPU);
		ImplicitSweep_3D_GPU_lineDivide(sf_d, uzGPU);

		Boundary_3D_GPU(roGPU);
		Boundary_3D_GPU(uxGPU);
		Boundary_3D_GPU(uyGPU);
		Boundary_3D_GPU(uzGPU);
	}
}