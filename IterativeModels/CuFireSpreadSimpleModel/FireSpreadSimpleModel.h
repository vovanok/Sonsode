#pragma once

#include <functional>
#include "IterativeModel.h"
#include "SonsodeFunctionsLib.cu"
#include "HostDataPrinter.hpp"
#include "SweepFactors.cu"
#include "HostData.hpp"
#include "DeviceData.cu"
#include "Vectors.hpp"
#include "FireSpreadSimpleModelPodTypes.cu"

#define BLOCK_SIZE_FSSM 4

using Sonsode::HostData1D;
using Sonsode::HostData2D;
using Sonsode::DeviceData1D;
using Sonsode::DeviceData2D;
using Sonsode::SweepFactors;
using Sonsode::FunctionsLib::ExplicitGaussSeidel_2D_CPU;
using Sonsode::FunctionsLib::ExplicitGaussSeidel_2D_GPU_direct;
using Sonsode::FunctionsLib::ImplicitSweep_2D_CPU;
using Sonsode::FunctionsLib::ImplicitSweep_2D_GPU_lineDivide;

class FireSpreadSimpleModel : public IterativeModel {
public:
	virtual std::string PrintData() const;
	virtual void SynchronizeWithGpu();

	FireSpreadSimpleModel(FireSpreadConsts consts, FireSpreadDataH data);
	virtual ~FireSpreadSimpleModel() {
		GpuOff();
		_data.Erase();
		lx.Erase();
		ly.Erase();
	}

protected:
	FireSpreadConsts _consts;

	FireSpreadDataH _data;
	FireSpreadDataD _data_dev;
	
	float hh;
	Vector2D<float> u; //Вспом перем = windU * cos(windAngle) и windU * sin(windAngle)
	float ap; //Вспом перем = m2/hh
	float r0; //Вспом перем = 1/h

	HostData1D<float> lx;
	HostData1D<float> ly;

	DeviceData1D<float> lx_dev;
	DeviceData1D<float> ly_dev;

	void CalculationMethod_CPU();
	void CalculationMethod_GPU();

	virtual void PrepareDataForGpu(const Sonsode::GpuDevice &gpu, size_t orderNumber);
	virtual void FreeDataForGpus();

	//Запуск ядра вычиления противоточных производных
	void Run_Kernel_FireSpreadSimpleModel_CounterflowDerivative(FireSpreadDataD data_dev,
		Vector2D<float> u, float r0, float danu, float hh);

	//Запуск ядра прогонки вдоль оси X
	void Run_Kernel_FireSpreadSimpleModel_RunAroundX(FireSpreadDataD data_dev,
		DeviceData1D<float> ly, float ap, float tau);

	//Запуск ядра прогонки вдоль оси Y
	void Run_Kernel_FireSpreadSimpleModel_RunAroundY(FireSpreadDataD data_dev,
		DeviceData1D<float> lx, float ap, float tau);

	//Запуск ядра вычисления температур после горения
	void Run_Kernel_FireSpreadSimpleModel_Fire(FireSpreadDataD data_dev,
		float temKr, float qbig, int mstep, float tzv, float tau, float qlitl, float ks, float vlag);
};