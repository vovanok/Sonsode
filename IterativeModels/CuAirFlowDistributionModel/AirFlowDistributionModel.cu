#include "AirFlowDistributionModel.h"

AirFlowDistributionModel::AirFlowDistributionModel(AirFlowConsts consts, AirFlowDataH data)
		: IterativeModel(consts.Tau), _consts(consts), _data(data) {

	sf_h = HostData3D<SweepFactors<float>>(data.dimX(), data.dimY(), data.dimZ());

	roCPU = AirFlowDistributionFunctor::Ro<AirFlowDataH>(_consts, _data);
	uxCPU = AirFlowDistributionFunctor::Ux<AirFlowDataH>(_consts, _data);
	uyCPU = AirFlowDistributionFunctor::Uy<AirFlowDataH>(_consts, _data);
	uzCPU = AirFlowDistributionFunctor::Uz<AirFlowDataH>(_consts, _data);

	AddCalculationMethod("cpu", std::bind(std::mem_fun(&AirFlowDistributionModel::CalculationMethod_CPU), this));
	AddCalculationMethod("gpu", std::bind(std::mem_fun(&AirFlowDistributionModel::CalculationMethod_GPU), this));
}

AirFlowDistributionModel::~AirFlowDistributionModel() {
	GpuOff();

	_data.Erase();
	sf_h.Erase();
}

std::string AirFlowDistributionModel::PrintData() const {
	return "";//HostDataPrinter::Print<AirFlowPoint>(md->Data, AirFlowDistributionModelData::PrintAirFlowPoint);
}

void AirFlowDistributionModel::SynchronizeWithGpu() {
	if (isGpuOn())
		_data_dev.PutTo(_data);
}

void AirFlowDistributionModel::PrepareDataForGpu(const GpuDevice& gpuDevice, size_t orderNumber) {
	sf_d = DeviceData3D<SweepFactors<float>>(gpuDevice, sf_h);
	_data_dev = AirFlowDataD(gpuDevice, _data);

	roGPU = AirFlowDistributionFunctor::Ro<AirFlowDataD>(_consts, _data_dev);
	uxGPU = AirFlowDistributionFunctor::Ux<AirFlowDataD>(_consts, _data_dev);
	uyGPU = AirFlowDistributionFunctor::Uy<AirFlowDataD>(_consts, _data_dev);
	uzGPU = AirFlowDistributionFunctor::Uz<AirFlowDataD>(_consts, _data_dev);
}

void AirFlowDistributionModel::FreeDataForGpus() {
	_data_dev.Erase();
	sf_d.Erase();
}

void AirFlowDistributionModel::CalculationMethod_CPU() {
	ImplicitSweep_3D_CPU(sf_h, roCPU);
	ImplicitSweep_3D_CPU(sf_h, uxCPU);
	ImplicitSweep_3D_CPU(sf_h, uyCPU);
	ImplicitSweep_3D_CPU(sf_h, uzCPU);

	Boundary_3D_CPU(roCPU);
	Boundary_3D_CPU(uxCPU);
	Boundary_3D_CPU(uyCPU);
	Boundary_3D_CPU(uzCPU);
}

void AirFlowDistributionModel::CalculationMethod_GPU() {
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