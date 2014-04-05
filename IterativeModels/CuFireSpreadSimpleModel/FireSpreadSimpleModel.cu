#include "FireSpreadSimpleModel.h"

FireSpreadSimpleModel::FireSpreadSimpleModel(FireSpreadConsts consts, FireSpreadDataH data)
		: IterativeModel(consts.Tau), _consts(consts), _data(data) {

	for(size_t y = 0; y < data.dimY(); y++) {
		data.t(0, y) = consts.TemOnBounds;
		data.t(data.dimX() - 1, y) = consts.TemOnBounds;
	}

	for (size_t x = 0; x < data.dimX(); x++) {
		data.t(x, 0) = consts.TemOnBounds;
		data.t(x, data.dimY() - 1) = consts.TemOnBounds;
	}

	sf_h = HostData2D<SweepFactors<float>>(data.dimX(), data.dimY());

	t4CPU = FireSpreadFunctor::T4<FireSpreadDataH>(consts, data);
	gorenieCPU = FireSpreadFunctor::Gorenie<FireSpreadDataH>(consts, data);
	temperatureCPU = FireSpreadFunctor::Temperature<FireSpreadDataH>(consts, data);

	AddCalculationMethod("cpu", std::bind(std::mem_fun(&FireSpreadSimpleModel::CalculationMethod_CPU), this));
	AddCalculationMethod("gpu", std::bind(std::mem_fun(&FireSpreadSimpleModel::CalculationMethod_GPU), this));
}

FireSpreadSimpleModel::~FireSpreadSimpleModel() {
	GpuOff();

	_data.Erase();
	sf_h.Erase();
}

std::string FireSpreadSimpleModel::PrintData() const {
	return "";
}

void FireSpreadSimpleModel::SynchronizeWithGpu() {
	if (isGpuOn())
		_data_dev.PutTo(_data);
}

void FireSpreadSimpleModel::PrepareDataForGpu(const Sonsode::GpuDevice &gpuDevice, size_t orderNumber) {
	sf_d = DeviceData2D<SweepFactors<float>>(gpuDevice, sf_h);
	_data_dev = FireSpreadDataD(gpuDevice, _data);

	t4GPU = FireSpreadFunctor::T4<FireSpreadDataD>(_consts, _data_dev);
	gorenieGPU = FireSpreadFunctor::Gorenie<FireSpreadDataD>(_consts, _data_dev);
	temperatureGPU = FireSpreadFunctor::Temperature<FireSpreadDataD>(_consts, _data_dev);
}

void FireSpreadSimpleModel::FreeDataForGpus() {
	_data_dev.Erase();
	sf_d.Erase();
}

void FireSpreadSimpleModel::CalculationMethod_CPU() {
	GpuOff();

	//Противоточные произодные
	ExplicitGaussSeidel_2D_CPU(t4CPU);
	
	//Новые температуры
	ImplicitSweep_2D_CPU(sf_h, temperatureCPU);

	//Горение
	if (currentIteration() >= _consts.IterFireBeginNum)
		FullSearch_2D_CPU(gorenieCPU);
}

void FireSpreadSimpleModel::CalculationMethod_GPU() {
	GpuOn();

	//Противоточные производные
	ExplicitGaussSeidel_2D_GPU_direct(t4GPU);

	//Новые температуры
	ImplicitSweep_2D_GPU_lineDivide(sf_d, temperatureGPU);

	//Горение
	if (currentIteration() >= _consts.IterFireBeginNum)
		FullSearch_2D_GPU(gorenieGPU);
}