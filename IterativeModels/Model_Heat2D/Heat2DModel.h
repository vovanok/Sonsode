#pragma once

#include <functional>
#include "IterativeModel.h"
#include "Heat2DFunctors.cu"
#include "SonsodeFunctionsLib.cu"
#include "HostDataPrinter.hpp"
#include "SweepFactors.cu"
#include "HostData.hpp"
#include "DeviceData.cu"

namespace Heat2D {
	using Sonsode::GpuDevice;
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
	using namespace Functors;

	class Heat2DModel : public IterativeModel {
	protected:
		HostData2D<float> _t; //Поле температур на хосте
		DeviceData2D<float> _t_dev; //Поле температур на устройстве
		float _h; //Шаг по расстоянию
		float _a; //Константа, характеризующая среду

		bool isSweepCpuInit;
		bool isSweepGpuInit;
		HostData2D<SweepFactors<float>> sf_h;
		DeviceData2D<SweepFactors<float>> sf_d;

		Heat2DFunctor<HostData2D<float>> fnCPU;
		Heat2DFunctor<DeviceData2D<float>> fnGPU;

		//BaseModel
		virtual void PrepareDataForGpu(const GpuDevice &gpu, size_t orderNumber);
		virtual void FreeDataForGpus();

		void CalculationMethod_CPU_GaussSeidel();
		void CalculationMethod_CPU_Sweep();
		void CalculationMethod_GPU_GaussSeidel_Direct();
		void CalculationMethod_GPU_GaussSeidel_Chess();
		void CalculationMethod_GPU_GaussSeidel_WithoutConflicts();
		void CalculationMethod_GPU_GaussSeidel_DirectOverlay();
		void CalculationMethod_GPU_Sweep_LineDevide();
		void CalculationMethod_GPU_Sweep_BlockDevide();

		void InitSweep(bool useGpu);
		void DeinitSweep();
	public:
		Heat2DModel(HostData2D<float> t, float h, float a, float tau);
		~Heat2DModel() {
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
}