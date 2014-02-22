#include "OilSpillageImprovedModel.h"

OilSpillageImprovedModel::OilSpillageImprovedModel(OilSpillageConsts consts, OilSpillageDataH data)
		: IterativeModel(consts.Tau), _data(data), _consts(consts) {
	data.FillS(0.0f);
	sf_h = HostData2D<SweepFactors<float>>(data.dimX(), data.dimY());

	waterUxCPU = OilSpillageImprovedFunctor::WaterUx<OilSpillageDataH>(consts, data);
	waterUyCPU = OilSpillageImprovedFunctor::WaterUy<OilSpillageDataH>(consts, data);
	impurityCPU = OilSpillageImprovedFunctor::Impurity<OilSpillageDataH>(consts, data);
	pressCPU = OilSpillageImprovedFunctor::Press<OilSpillageDataH>(consts, data);
	waterUxSCPU = OilSpillageImprovedFunctor::WaterUxS<OilSpillageDataH>(consts, data);
	waterUySCPU = OilSpillageImprovedFunctor::WaterUyS<OilSpillageDataH>(consts, data);
	waterUx_CompleteCPU = OilSpillageImprovedFunctor::WaterUx_Complete<OilSpillageDataH>(consts, data);
	waterUy_CompleteCPU = OilSpillageImprovedFunctor::WaterUy_Complete<OilSpillageDataH>(consts, data);
	oilUxCPU = OilSpillageImprovedFunctor::OilUx<OilSpillageDataH>(consts, data);
	oilUyCPU = OilSpillageImprovedFunctor::OilUy<OilSpillageDataH>(consts, data);
	pressSCPU = OilSpillageImprovedFunctor::PressS<OilSpillageDataH>(consts, data);
	wCPU = OilSpillageImprovedFunctor::W<OilSpillageDataH>(consts, data);
	impurityIstokCPU = OilSpillageImprovedFunctor::ImpurityIstok<OilSpillageDataH>(consts, data);
	impuritySCPU = OilSpillageImprovedFunctor::ImpurityS<OilSpillageDataH>(consts, data);
	islandResolverCPU = OilSpillageImprovedFunctor::IslandResolver<OilSpillageDataH>(consts, data);

	AddCalculationMethod("cpu", std::bind(std::mem_fun(&OilSpillageImprovedModel::CalculationMethod_CPU), this));
	AddCalculationMethod("gpu", std::bind(std::mem_fun(&OilSpillageImprovedModel::CalculationMethod_GPU), this));
}

OilSpillageImprovedModel::~OilSpillageImprovedModel() {
	GpuOff();

	_data.Erase();
	sf_h.Erase();
}

std::string OilSpillageImprovedModel::PrintData() const {
	return "";
}

void OilSpillageImprovedModel::SynchronizeWithGpu() {
	if (isGpuOn())
		_data_dev.PutTo(_data);
}

void OilSpillageImprovedModel::PrepareDataForGpu(const Sonsode::GpuDevice &gpuDevice, size_t orderNumber) {
	sf_d = DeviceData2D<SweepFactors<float>>(gpuDevice, sf_h);
	_data_dev = OilSpillageDataD(gpuDevice, _data);

	waterUxGPU = OilSpillageImprovedFunctor::WaterUx<OilSpillageDataD>(_consts, _data_dev);
	waterUyGPU = OilSpillageImprovedFunctor::WaterUy<OilSpillageDataD>(_consts, _data_dev);
	impurityGPU = OilSpillageImprovedFunctor::Impurity<OilSpillageDataD>(_consts, _data_dev);
	pressGPU = OilSpillageImprovedFunctor::Press<OilSpillageDataD>(_consts, _data_dev);
	waterUxSGPU = OilSpillageImprovedFunctor::WaterUxS<OilSpillageDataD>(_consts, _data_dev);
	waterUySGPU = OilSpillageImprovedFunctor::WaterUyS<OilSpillageDataD>(_consts, _data_dev);
	waterUx_CompleteGPU = OilSpillageImprovedFunctor::WaterUx_Complete<OilSpillageDataD>(_consts, _data_dev);
	waterUy_CompleteGPU = OilSpillageImprovedFunctor::WaterUy_Complete<OilSpillageDataD>(_consts, _data_dev);
	oilUxGPU = OilSpillageImprovedFunctor::OilUx<OilSpillageDataD>(_consts, _data_dev);
	oilUyGPU = OilSpillageImprovedFunctor::OilUy<OilSpillageDataD>(_consts, _data_dev);
	pressSGPU = OilSpillageImprovedFunctor::PressS<OilSpillageDataD>(_consts, _data_dev);
	wGPU = OilSpillageImprovedFunctor::W<OilSpillageDataD>(_consts, _data_dev);
	impurityIstokGPU = OilSpillageImprovedFunctor::ImpurityIstok<OilSpillageDataD>(_consts, _data_dev);
	impuritySGPU = OilSpillageImprovedFunctor::ImpurityS<OilSpillageDataD>(_consts, _data_dev);
	islandResolverGPU = OilSpillageImprovedFunctor::IslandResolver<OilSpillageDataD>(_consts, _data_dev);
}

void OilSpillageImprovedModel::FreeDataForGpus() {
	_data_dev.Erase();
	sf_d.Erase();
}

void OilSpillageImprovedModel::CalculationMethod_CPU() {
	GpuOff();
	
	//gran
	//ResolveBoundaryConditions();

	ImplicitSweep_2D_CPU(sf_h, waterUxCPU);
	ImplicitSweep_2D_CPU(sf_h, waterUyCPU);

	//CalculateS_Press
	ExplicitGaussSeidel_2D_CPU(pressSCPU);

	Boundary_2D_CPU(pressSCPU);
	Boundary_2D_CPU(impuritySCPU);

	ImplicitSweep_2D_CPU(sf_h, pressCPU);

	//Press boundary condition
	Boundary_2D_CPU(pressCPU);

	ExplicitGaussSeidel_2D_CPU(waterUxSCPU);
	ExplicitGaussSeidel_2D_CPU(waterUySCPU);

	ExplicitGaussSeidel_2D_CPU(waterUx_CompleteCPU);
	ExplicitGaussSeidel_2D_CPU(waterUy_CompleteCPU);

	ExplicitGaussSeidel_2D_CPU(oilUxCPU);
	ExplicitGaussSeidel_2D_CPU(oilUyCPU);

	//gran
	//ResolveBoundaryConditions();

	ExplicitGaussSeidel_2D_CPU(wCPU);
	ImplicitSweep_2D_CPU(sf_h, impurityCPU);

	//gran
	//ResolveBoundaryConditions();

	ExplicitGaussSeidel_2D_CPU(impurityIstokCPU);

	FullSearch_2D_CPU(islandResolverCPU);
}

void OilSpillageImprovedModel::CalculationMethod_GPU() {
	GpuOn();

	////gran
	////ResolveBoundaryConditions();
	
	ImplicitSweep_2D_GPU_lineDivide(sf_d, waterUxGPU);
	ImplicitSweep_2D_GPU_lineDivide(sf_d, waterUyGPU);

	////CalculateS_Press
	ExplicitGaussSeidel_2D_GPU_direct(pressSGPU);

	Boundary_2D_GPU(pressSGPU);
	Boundary_2D_GPU(impuritySGPU);

	ImplicitSweep_2D_GPU_lineDivide(sf_d, pressGPU);

	//Press boundary condition
	Boundary_2D_GPU(pressGPU);

	ExplicitGaussSeidel_2D_GPU_direct(waterUxSGPU);
	ExplicitGaussSeidel_2D_GPU_direct(waterUySGPU);

	ExplicitGaussSeidel_2D_GPU_direct(waterUx_CompleteGPU);
	ExplicitGaussSeidel_2D_GPU_direct(waterUy_CompleteGPU);

	ExplicitGaussSeidel_2D_GPU_direct(oilUxGPU);
	ExplicitGaussSeidel_2D_GPU_direct(oilUyGPU);

	////gran
	////ResolveBoundaryConditions();

	ExplicitGaussSeidel_2D_GPU_direct(wGPU);
	ImplicitSweep_2D_GPU_lineDivide(sf_d, impurityGPU);

	////gran
	////ResolveBoundaryConditions();

	ExplicitGaussSeidel_2D_GPU_direct(impurityIstokGPU);

	FullSearch_2D_GPU(islandResolverGPU);
}

void OilSpillageImprovedModel::ResolveBoundaryConditions() {
	/*for (size_t x = 0; x < dimX(); x++) {
		for (size_t y = 0; y < dimY(); y++) {

			if (deep(x, y) <= 0.0f) {
				waterUx(x, y) = 0.0f;
				waterUy(x, y) = 0.0f;

				oilUx(x, y) = 0.0f;
				oilUy(x, y) = 0.0f;

				if (x != 0)
					continue;

				impurity(1, y) = impurity(0, y);
				waterUx(0, y) = 0.3f;
				waterUy(0, y) = 0.0f;
				oilUx(0, y) = 0.3f;
				oilUy(0, y) = 0.0f;
			}
		}
	}*/
}