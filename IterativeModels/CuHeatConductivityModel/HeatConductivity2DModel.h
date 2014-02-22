#pragma once

#include <functional>
#include "IterativeModel.h"
#include "HeatConductivity2D_Functors.cu"
#include "SonsodeFunctionsLib.cu"
#include "HostDataPrinter.hpp"
#include "SweepFactors.cu"
#include "HostData.hpp"
#include "DeviceData.cu"

using Sonsode::HostData2D;
using Sonsode::DeviceData2D;
using Sonsode::SweepFactors;
using Sonsode::FunctionsLib::ExplicitGaussSeidel_2D_CPU;
using Sonsode::FunctionsLib::ExplicitGaussSeidel_2D_GPU_direct;
using Sonsode::FunctionsLib::ExplicitGaussSeidel_2D_GPU_chess;
using Sonsode::FunctionsLib::ExplicitGaussSeidel_2D_GPU_outconf;
using Sonsode::FunctionsLib::ExplicitGaussSeidel_2D_GPU_directOverlay;
using Sonsode::FunctionsLib::ImplicitSweep_2D_CPU;
using Sonsode::FunctionsLib::ImplicitSweep_2D_GPU_lineDivide;
using Sonsode::FunctionsLib::ImplicitSweep_2D_GPU_blockDivide;

class HeatConductivity2DModel : public IterativeModel {
protected:
	HostData2D<float> _t; //Поле температур на хосте
	DeviceData2D<float> _t_dev; //Поле температур на устройстве
	float _h; //Шаг по расстоянию
	float _a; //Константа, характеризующая среду

	bool isSweepCpuInit;
	bool isSweepGpuInit;
	HostData2D<SweepFactors<float>> sf_h;
	DeviceData2D<SweepFactors<float>> sf_d;

	HeatConductivity2DFunctor<HostData2D<float>> fnCPU;
	HeatConductivity2DFunctor<DeviceData2D<float>> fnGPU;

	//BaseModel
	virtual void PrepareDataForGpu(const Sonsode::GpuDevice &gpu, size_t orderNumber) throw(std::string);
	virtual void FreeDataForGpus() throw(std::string);

	void CalculationMethod_CPU_GaussSeidel();
	void CalculationMethod_CPU_Sweep();
	void CalculationMethod_GPU_GaussSeidel_Direct();
	void CalculationMethod_GPU_GaussSeidel_Chess();
	void CalculationMethod_GPU_GaussSeidel_WithoutConflicts();
	void CalculationMethod_GPU_GaussSeidel_DirectOverlay();
	void CalculationMethod_GPU_Sweep_LineDevide();
	void CalculationMethod_GPU_Sweep_BlockDevide();

	void InitSweep(bool useGpu) throw (std::string);
	void DeinitSweep() throw (std::string);
public:
	HeatConductivity2DModel(HostData2D<float> t, float h, float a, float tau);
	~HeatConductivity2DModel() {
		GpuOff();
		DeinitSweep();
		_t.Erase();
	}

	virtual std::string PrintData() const;
	virtual void SynchronizeWithGpu();

	float h() const { return _h; }
	float a() const { return _a; }
	HostData2D<float> t() { return _t; }
};