#pragma once

#include "HostData.hpp"
#include "DeviceData.cu"
#include "SweepFactors.cu"

namespace Sonsode {
	namespace Kernels {

		namespace {
			__host__ __device__ float GetNewValue(const float nextValue, const SweepFactors<float>& lm) {
				return lm.L * nextValue + lm.M;
			}
		}
		
		template<class FunctorType>
		void HostKernel_ImplicitSweep_2D_CPU(
			HostData2D<SweepFactors<float>> sweepFactors, FunctorType fn);

		template<class FunctorType>
		__global__ void Kernel_ImplicitSweepAroundX_2D_GPUlineDivide(
			DeviceData2D<SweepFactors<float>> sweepFactors, FunctorType fn);

		template<class FunctorType>
		__global__ void Kernel_ImplicitSweepAroundY_2D_GPUlineDivide(
			DeviceData2D<SweepFactors<float>> sweepFactors, FunctorType fn);

		template<class FunctorType>
		__global__ void Kernel_ImplicitSweep_2D_GPUblockDivide(
			DeviceData2D<SweepFactors<float>> sweepFactors, FunctorType fn);

	}
}

namespace Sonsode {
	namespace Kernels {
		#pragma region _2D

		template<class FunctorType>
		void HostKernel_ImplicitSweep_2D_CPU(
				HostData2D<SweepFactors<float>> sweepFactors, FunctorType fn) {
			
			//Around X
			for (size_t y = 1; y < fn.dimY() - 1; y++) {
				//Forward
				sweepFactors(0, y).L = fn.IsZero(0, y) ? 0.0f : 1.0f;
				sweepFactors(fn.dimX()-1, y).L = fn.IsZero(fn.dimX()-1, y) ? 0.0f : 1.0f;

				for (size_t x = 0; x < fn.dimX()-1; x++)
					sweepFactors(x+1, y) = sweepFactors(x, y).GetNextFactors(fn.QX(x, y), fn.AlphaX(x, y), fn.BetaX(x, y), fn.GammaX(x, y));
			
				//Reverse
				for (size_t x = fn.dimX()-2; x >= 1; x--)
					fn.setValue(x, y, GetNewValue(fn.getValue(x+1, y), sweepFactors(x+1, y)));
			}

			//Around Y
			for (size_t x = 1; x < fn.dimX()-1; x++) {
				//Forward
				sweepFactors(x, 0).L = fn.IsZero(x, 0) ? 0.0f : 1.0f;
				sweepFactors(x, fn.dimY()-1).L = fn.IsZero(x, fn.dimY()-1) ? 0.0f : 1.0f;

				for (size_t y = 0; y < fn.dimY()-1; y++)
					sweepFactors(x, y+1) = sweepFactors(x, y).GetNextFactors(fn.QY(x, y), fn.AlphaY(x, y), fn.BetaY(x, y), fn.GammaY(x, y));

				//Reverse
				for (size_t y = fn.dimY()-2; y >= 1; y--)
					fn.setValue(x, y, GetNewValue(fn.getValue(x, y+1), sweepFactors(x, y+1)));
			}
		}

		template<class FunctorType>
		__global__ void Kernel_ImplicitSweepAroundX_2D_GPUlineDivide(
				DeviceData2D<SweepFactors<float>> sweepFactors, FunctorType fn) {
			
			size_t y = blockIdx.x*blockDim.x+threadIdx.x;

			if (y > 0 && y < fn.dimY()-1) {
				//Прямая прогонка
				sweepFactors(0, y).L = fn.IsZero(0, y) ? 0.0f : 1.0f;
				sweepFactors(fn.dimX()-1, y).L = fn.IsZero(fn.dimX()-1, y) ? 0.0f : 1.0f;

				for (size_t x = 0; x < fn.dimX()-1; x++)
					sweepFactors(x+1, y) = sweepFactors(x, y).GetNextFactors(fn.QX(x, y), fn.AlphaX(x, y), fn.BetaX(x, y), fn.GammaX(x, y));
			
				//Обратная прогонка
				for (size_t x = fn.dimX() - 2; x >= 1; x--)
					fn.setValue(x, y, GetNewValue(fn.getValue(x + 1, y), sweepFactors(x + 1, y)));
			}
		}

		template<class FunctorType>
		__global__ void Kernel_ImplicitSweepAroundY_2D_GPUlineDivide(
				DeviceData2D<SweepFactors<float>> sweepFactors, FunctorType fn) {
			
			size_t x = blockIdx.x*blockDim.x+threadIdx.x;

			if (x > 0 && x < fn.dimX()-1) {
				//Прямая пронка
				sweepFactors(x, 0).L = fn.IsZero(x, 0) ? 0.0f : 1.0f;
				sweepFactors(x, fn.dimY()-1).L = fn.IsZero(x, fn.dimY()-1) ? 0.0f : 1.0f;

				for (size_t y = 0; y < fn.dimY()-1; y++)
					sweepFactors(x, y+1) = sweepFactors(x, y).GetNextFactors(fn.QY(x, y), fn.AlphaY(x, y), fn.BetaY(x, y), fn.GammaY(x, y));

				//Обратная прогонка
				for (size_t y = fn.dimY()-2; y >= 1; y--)
					fn.setValue(x, y, GetNewValue(fn.getValue(x, y+1), sweepFactors(x, y+1)));
			}
		}

		template<class FunctorType>
		__global__ void Kernel_ImplicitSweep_2D_GPUblockDivide(
				DeviceData2D<SweepFactors<float>> sweepFactors, FunctorType fn, size_t overlayValue) {

			//size_t x = blockIdx.x * (blockDim.x - (2 * overlayValue)) + threadIdx.x;
			//size_t y = blockIdx.y * (blockDim.y - (2 * overlayValue)) + threadIdx.y;
		}

		#pragma endregion

		#pragma region _3D
		
		template<class FunctorType>
		void HostKernel_ImplicitSweep_3D_CPU(
				HostData3D<SweepFactors<float>> sweepFactors, FunctorType fn) {

			//Around X
			for (size_t y = 1; y < fn.dimY()-1; y++) {
				for (size_t z = 1; z < fn.dimZ()-1; z++) {
					//Forward
					sweepFactors(0, y, z).L = fn.IsZero(0, y, z) ? 0.0f : 1.0f;
					sweepFactors(fn.dimX() - 1, y, z).L = fn.IsZero(fn.dimX() - 1, y, z) ? 0.0f : 1.0f;

					for (size_t x = 0; x < fn.dimX() - 1; x++)
						sweepFactors(x + 1, y, z) = sweepFactors(x, y, z).GetNextFactors(fn.QX(x, y, z), fn.AlphaX(x, y, z), fn.BetaX(x, y, z), fn.GammaX(x, y, z));
			
					//Reverse
					for (size_t x = fn.dimX() - 2; x >= 1; x--)
						fn.setValue(x, y, z, GetNewValue(fn.getValue(x + 1, y, z), sweepFactors(x + 1, y, z)));
				}
			}

			//Around Y
			for (size_t x = 1; x < fn.dimX()-1; x++) {
				for (size_t z = 1; z < fn.dimZ()-1; z++) {
					//Forward
					sweepFactors(x, 0, z).L = fn.IsZero(x, 0, z) ? 0.0f : 1.0f;
					sweepFactors(x, fn.dimY()-1, z).L = fn.IsZero(x, fn.dimY()-1, z) ? 0.0f : 1.0f;

					for (size_t y = 0; y < fn.dimY()-1; y++)
						sweepFactors(x, y+1, z) = sweepFactors(x, y, z).GetNextFactors(fn.QY(x, y, z), fn.AlphaY(x, y, z), fn.BetaY(x, y, z), fn.GammaY(x, y, z));

					//Reverse
					for (size_t y = fn.dimY()-2; y >= 1; y--)
						fn.setValue(x, y, z, GetNewValue(fn.getValue(x, y+1, z), sweepFactors(x, y+1, z)));
				}
			}

			//Around Z
			for (size_t x = 1; x < fn.dimX()-1; x++) {
				for (size_t y = 1; y < fn.dimY()-1; y++) {
					//Forward
					sweepFactors(x, y, 0).L = fn.IsZero(x, y, 0) ? 0.0f : 1.0f;
					sweepFactors(x, y, fn.dimZ()-1).L = fn.IsZero(x, y, fn.dimZ()-1) ? 0.0f : 1.0f;

					for (size_t z = 0; z < fn.dimZ()-1; z++)
						sweepFactors(x, y, z + 1) = sweepFactors(x, y, z).GetNextFactors(fn.QY(x, y, z), fn.AlphaY(x, y, z), fn.BetaY(x, y, z), fn.GammaY(x, y, z));

					//Reverse
					for (size_t z = fn.dimZ()-2; z >= 1; z--)
						fn.setValue(x, y, z, GetNewValue(fn.getValue(x, y, z+1), sweepFactors(x, y, z+1)));
				}
			}
		}

		template<class FunctorType>
		__global__ void Kernel_ImplicitSweepAroundX_3D_GPUlineDivide(
				DeviceData3D<SweepFactors<float>> sweepFactors, FunctorType fn) {

			size_t y = blockIdx.x * blockDim.x + threadIdx.x;
			size_t z = blockIdx.y * blockDim.y + threadIdx.y;

			if (y == 0 || z == 0 || y > fn.dimY() - 2 || z > fn.dimZ() - 2)
				return;

			//Прямая прогонка
			sweepFactors(0, y, z).L = fn.IsZero(0, y, z) ? 0.0f : 1.0f;
			sweepFactors(fn.dimX()-1, y, z).L = fn.IsZero(fn.dimX()-1, y, z) ? 0.0f : 1.0f;

			for (size_t x = 0; x < fn.dimX()-1; x++)
				sweepFactors(x+1, y, z) = sweepFactors(x, y, z).GetNextFactors(fn.QX(x, y, z), fn.AlphaX(x, y, z), fn.BetaX(x, y, z), fn.GammaX(x, y, z));
			
			//Обратная прогонка
			for (size_t x = fn.dimX()-2; x >= 1; x--)
				fn.setValue(x, y, z, GetNewValue(fn.getValue(x+1, y, z), sweepFactors(x+1, y, z)));
		}

		template<class FunctorType>
		__global__ void Kernel_ImplicitSweepAroundY_3D_GPUlineDivide(
				DeviceData3D<SweepFactors<float>> sweepFactors, FunctorType fn) {

			size_t x = blockIdx.x * blockDim.x + threadIdx.x;
			size_t z = blockIdx.y * blockDim.y + threadIdx.y;

			if (x == 0 || z == 0 || x > fn.dimX() - 2 || z > fn.dimZ() - 2)
				return;

			//Прямая прогонка
			sweepFactors(x, 0, z).L = fn.IsZero(x, 0, z) ? 0.0f : 1.0f;
			sweepFactors(x, fn.dimY()-1, z).L = fn.IsZero(x, fn.dimY()-1, z) ? 0.0f : 1.0f;

			for (size_t y = 0; y < fn.dimY()-1; y++)
				sweepFactors(x, y+1, z) = sweepFactors(x, y, z).GetNextFactors(fn.QX(x, y, z), fn.AlphaX(x, y, z), fn.BetaX(x, y, z), fn.GammaX(x, y, z));
			
			//Обратная прогонка
			for (size_t y = fn.dimY()-2; y >= 1; y--)
				fn.setValue(x, y, z, GetNewValue(fn.getValue(x, y+1, z), sweepFactors(x, y+1, z)));
		}

		template<class FunctorType>
		__global__ void Kernel_ImplicitSweepAroundZ_3D_GPUlineDivide(
				DeviceData3D<SweepFactors<float>> sweepFactors, FunctorType fn) {

			size_t x = blockIdx.x * blockDim.x + threadIdx.x;
			size_t y = blockIdx.y * blockDim.y + threadIdx.y;

			if (x == 0 || y == 0 || x > fn.dimX()-2 || y > fn.dimY()-2)
				return;

			//Прямая прогонка
			sweepFactors(x, y, 0).L = fn.IsZero(x, y, 0) ? 0.0f : 1.0f;
			sweepFactors(x, y, fn.dimZ()-1).L = fn.IsZero(x, y, fn.dimZ()-1) ? 0.0f : 1.0f;

			for (size_t z = 0; z < fn.dimZ()-1; z++)
				sweepFactors(x, y, z+1) = sweepFactors(x, y, z).GetNextFactors(fn.QX(x, y, z), fn.AlphaX(x, y, z), fn.BetaX(x, y, z), fn.GammaX(x, y, z));
			
			//Обратная прогонка
			for (size_t z = fn.dimZ()-2; z >= 1; z--)
				fn.setValue(x, y, z, GetNewValue(fn.getValue(x, y, z+1), sweepFactors(x, y, z+1)));
		}

		template<class FunctorType>
		__global__ void Kernel_ImplicitSweep_3D_GPUblockDivide(
				DeviceData2D<SweepFactors<float>> sweepFactors, FunctorType fn) {
		}

		#pragma endregion
	}
}