#pragma once

#include "SonsodeCommon.h"
#include "Kernels_ExplicitGaussSeidel.cu"
#include "Kernels_ImplicitSweep.cu"
#include "Kernels_BoundaryConditions.cu"
#include <math.h>

namespace {
	size_t GetBlockCount(size_t dataSize) {
		return (dataSize / BLOCK_SIZE) + ((dataSize % BLOCK_SIZE) > 0 ? 1 : 0);
	}

	size_t GetBlockCount(size_t dataSize, size_t overlayValue) {
		return ((dataSize - overlayValue * 2) / (BLOCK_SIZE - overlayValue * 2)) +
			(((dataSize - overlayValue * 2) % (BLOCK_SIZE - overlayValue * 2)) > 0 ? 1 : 0);
	}
}

namespace Sonsode {
	namespace FunctionsLib {
		template<class FunctorType> void ExplicitGaussSeidel_2D_CPU(FunctorType fn);
		template<class FunctorType> void ExplicitGaussSeidel_2D_GPU_direct(FunctorType fn);
		template<class FunctorType> void ExplicitGaussSeidel_2D_GPU_chess(FunctorType fn);
		template<class FunctorType> void ExplicitGaussSeidel_2D_GPU_outconf(FunctorType fn);
		template<class FunctorType> void ExplicitGaussSeidel_2D_GPU_directOverlay(FunctorType fn);

		template<class FunctorType> void ExplicitGaussSeidel_3D_CPU(FunctorType fn);
		template<class FunctorType> void ExplicitGaussSeidel_3D_GPU_direct(FunctorType fn);

		template<class FunctorType> void ImplicitSweep_2D_CPU(HostData2D<SweepFactors<float>> sweepFactors, FunctorType fn);
		template<class FunctorType> void ImplicitSweep_2D_GPU_lineDivide(DeviceData2D<SweepFactors<float>> sweepFactors, FunctorType fn);
		template<class FunctorType> void ImplicitSweep_2D_GPU_blockDivide(DeviceData2D<SweepFactors<float>> sweepFactors, FunctorType fn);

		template<class FunctorType> void ImplicitSweep_3D_CPU(HostData3D<SweepFactors<float>> sweepFactors, FunctorType fn);
		template<class FunctorType> void ImplicitSweep_3D_GPU_lineDivide(DeviceData3D<SweepFactors<float>> sweepFactors, FunctorType fn);
		template<class FunctorType> void ImplicitSweep_3D_GPU_blockDivide(DeviceData3D<SweepFactors<float>> sweepFactors, FunctorType fn);

		template<class FunctorType> void Boundary_2D_CPU(FunctorType fn);
		template<class FunctorType> void Boundary_2D_GPU(FunctorType fn);

		template<class FunctorType> void Boundary_3D_CPU(FunctorType fn);
		template<class FunctorType> void Boundary_3D_GPU(FunctorType fn);

		template<class FunctorType> void FullSearch_2D_CPU(FunctorType fn);
		template<class FunctorType> void FullSearch_2D_GPU(FunctorType fn);
	}
}

namespace Sonsode {
	namespace FunctionsLib {
		#pragma region Explicit schemes

		#pragma region _2D

		template<class FunctorType>
		void ExplicitGaussSeidel_2D_CPU(FunctorType fn) {
			Sonsode::Kernels::HostKernel_ExplicitGaussSeidel_2D(fn);
		}

		template<class FunctorType>
		void ExplicitGaussSeidel_2D_GPU_direct(FunctorType fn) {
			size_t gridSizeX = GetBlockCount(fn.dimX());
			size_t gridSizeY = GetBlockCount(fn.dimY());

			dim3 threads (BLOCK_SIZE, BLOCK_SIZE);
			dim3 blocks (gridSizeX, gridSizeY);
	
			fn.gpu().SetAsCurrent();
			Sonsode::Kernels::Kernel_ExplicitGaussSeidel_2D_Direct<<<blocks, threads>>>(fn);
			fn.gpu().Synchronize();
		}

		template<class FunctorType>
		void ExplicitGaussSeidel_2D_GPU_chess(FunctorType fn) {
			size_t gridSizeX = GetBlockCount(fn.dimX());
			size_t gridSizeY = GetBlockCount(fn.dimY());

			dim3 threads (BLOCK_SIZE, BLOCK_SIZE);
			dim3 blocks (gridSizeX, gridSizeY);
	
			fn.gpu().SetAsCurrent();
			Sonsode::Kernels::Kernel_ExplicitGaussSeidel_2D_Chess<<<blocks, threads>>>(fn);
			fn.gpu().Synchronize();
		}

		template<class FunctorType>
		void ExplicitGaussSeidel_2D_GPU_outconf(FunctorType fn) {
			size_t gridSizeX = GetBlockCount(fn.dimX());
			size_t gridSizeY = GetBlockCount(fn.dimY());

			dim3 threads (BLOCK_SIZE, BLOCK_SIZE);
			dim3 blocks (gridSizeX, gridSizeY);
	
			fn.gpu().SetAsCurrent();
			Sonsode::Kernels::Kernel_ExplicitGaussSeidel_2D_OutConf<<<blocks, threads>>>(fn);
			fn.gpu().Synchronize();
		}

		template<class FunctorType>
		void ExplicitGaussSeidel_2D_GPU_directOverlay(FunctorType fn) {
			size_t overlayValue = 2;

			size_t gridSizeX = GetBlockCount(fn.dimX(), overlayValue);
			size_t gridSizeY = GetBlockCount(fn.dimY(), overlayValue);

			dim3 threads (BLOCK_SIZE, BLOCK_SIZE);
			dim3 blocks (gridSizeX, gridSizeY);
	
			fn.gpu().SetAsCurrent();
			Sonsode::Kernels::Kernel_ExplicitGaussSeidel_2D_DirectOverlay<<<blocks, threads>>>(fn, overlayValue);
			fn.gpu().Synchronize();
		}
		#pragma endregion

		#pragma region _3D

		template<class FunctorType>
		void ExplicitGaussSeidel_3D_CPU(FunctorType fn) {
			Sonsode::Kernels::HostKernel_ExplicitGaussSeidel_3D(fn);
		}
		
		template<class FunctorType>
		void ExplicitGaussSeidel_3D_GPU_direct(FunctorType fn) {
			//Виртуальный размер решетки
			int gridDimX = GetBlockCount(fn.dimX());
			int gridDimY = GetBlockCount(fn.dimY());
			int gridDimZ = GetBlockCount(fn.dimZ());

			//Фактический размер решетки
			int gridSize = gridDimX * gridDimY * gridDimZ;

			dim3 threads(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
			dim3 blocks(gridSize);

			fn.gpu().SetAsCurrent();
			Sonsode::Kernels::Kernel_ExplicitGaussSeidel_3D_Direct<<<blocks, threads>>>(fn, gridDimX, gridDimY, gridDimZ);
			fn.gpu().Synchronize();
		}

		#pragma endregion
	
		#pragma endregion

		#pragma region Implicit schemes

		#pragma region _2D

		template<class FunctorType>
		void ImplicitSweep_2D_CPU(HostData2D<SweepFactors<float>> sweepFactors, FunctorType fn) {
			Sonsode::Kernels::HostKernel_ImplicitSweep_2D_CPU(sweepFactors, fn);
		}

		template<class FunctorType>
		void ImplicitSweep_2D_GPU_lineDivide(DeviceData2D<SweepFactors<float>> sweepFactors, FunctorType fn) {
			size_t gridSize;
			
			//Прогонка по X
			gridSize = GetBlockCount(fn.dimY());
			fn.gpu().SetAsCurrent();
			Sonsode::Kernels::Kernel_ImplicitSweepAroundX_2D_GPUlineDivide<<<gridSize, BLOCK_SIZE>>>(sweepFactors, fn);
			fn.gpu().Synchronize();
			fn.gpu().CheckLastErr();

			//Прогонка по Y
			gridSize = GetBlockCount(fn.dimX());
			fn.gpu().SetAsCurrent();
			Sonsode::Kernels::Kernel_ImplicitSweepAroundY_2D_GPUlineDivide<<<gridSize, BLOCK_SIZE>>>(sweepFactors, fn);
			fn.gpu().Synchronize();
			fn.gpu().CheckLastErr();
		}

		template<class FunctorType>
		void ImplicitSweep_2D_GPU_blockDivide(DeviceData2D<SweepFactors<float>> sweepFactors, FunctorType fn) {
			const size_t overlayValue = 1;

			size_t gridSizeX = GetBlockCount(fn.dimX(), overlayValue);
			size_t gridSizeY = GetBlockCount(fn.dimY(), overlayValue);

			dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
			dim3 blocks(gridSizeX, gridSizeY);

			fn.gpu().SetAsCurrent();
			Sonsode::Kernels::Kernel_ImplicitSweep_2D_GPUblockDivide<<<blocks, threads>>>(sweepFactors, fn, overlayValue);
			fn.gpu().Synchronize();
		}

		#pragma endregion

		#pragma region _3D

		template<class FunctorType>
		void ImplicitSweep_3D_CPU(HostData3D<SweepFactors<float>> sweepFactors, FunctorType fn) {
			Sonsode::Kernels::HostKernel_ImplicitSweep_3D_CPU(sweepFactors, fn);
		}
		
		template<class FunctorType>
		void ImplicitSweep_3D_GPU_lineDivide(DeviceData3D<SweepFactors<float>> sweepFactors, FunctorType fn) {
			size_t gridSizeX;
			size_t gridSizeY;
			dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
			
			//Прогонка по X
			gridSizeX = GetBlockCount(fn.dimY());
			gridSizeY = GetBlockCount(fn.dimZ());
			
			fn.gpu().SetAsCurrent();
			Sonsode::Kernels::Kernel_ImplicitSweepAroundX_3D_GPUlineDivide<<<dim3(gridSizeX, gridSizeY), threads>>>(sweepFactors, fn);
			fn.gpu().Synchronize();

			//Прогонка по Y
			gridSizeX = GetBlockCount(fn.dimX());
			gridSizeY = GetBlockCount(fn.dimZ());

			fn.gpu().SetAsCurrent();
			Sonsode::Kernels::Kernel_ImplicitSweepAroundY_3D_GPUlineDivide<<<dim3(gridSizeX, gridSizeY), threads>>>(sweepFactors, fn);
			fn.gpu().Synchronize();

			//Прогонка по Z
			gridSizeX = GetBlockCount(fn.dimX());
			gridSizeY = GetBlockCount(fn.dimY());

			fn.gpu().SetAsCurrent();
			Sonsode::Kernels::Kernel_ImplicitSweepAroundZ_3D_GPUlineDivide<<<dim3(gridSizeX, gridSizeY), threads>>>(sweepFactors, fn);
			fn.gpu().Synchronize();
		}
		
		template<class FunctorType>
		void ImplicitSweep_3D_GPU_blockDivide(DeviceData3D<SweepFactors<float>> sweepFactors, FunctorType fn) {
			throw std::exception("Not implemented");
		}

		#pragma endregion

		#pragma endregion

		#pragma region Boundary

		template<class FunctorType>
		void Boundary_2D_CPU(FunctorType fn) {
			Sonsode::Kernels::HostKernel_Boundary_2D(fn);
		}

		template<class FunctorType>
		void Boundary_2D_GPU(FunctorType fn) {
			dim3 threads (BLOCK_SIZE);
			dim3 blocks (GetBlockCount(max(fn.dimX(), fn.dimY())));
	
			fn.gpu().SetAsCurrent();
			Sonsode::Kernels::Kernel_Boundary_2D<<<blocks, threads>>>(fn);
			fn.gpu().Synchronize();
		}

		template<class FunctorType>
		void Boundary_3D_CPU(FunctorType fn) {
			Sonsode::Kernels::HostKernel_Boundary_3D(fn);
		}

		template<class FunctorType>
		void Boundary_3D_GPU(FunctorType fn) {
			dim3 threads (BLOCK_SIZE);

			size_t gridDim = GetBlockCount(max(max(fn.dimX(), fn.dimY()), fn.dimZ()));
			dim3 blocks (gridDim, gridDim);
	
			fn.gpu().SetAsCurrent();
			Sonsode::Kernels::Kernel_Boundary_3D<<<blocks, threads>>>(fn);
			fn.gpu().Synchronize();
		}

		#pragma endregion

		#pragma region Full search

		template<class FunctorType>
		void FullSearch_2D_CPU(FunctorType fn) {
			Sonsode::Kernels::HostKernel_FullSearch_2D(fn);
		}

		template<class FunctorType>
		void FullSearch_2D_GPU(FunctorType fn) {
			size_t gridSizeX = GetBlockCount(fn.dimX());
			size_t gridSizeY = GetBlockCount(fn.dimY());

			dim3 threads (BLOCK_SIZE, BLOCK_SIZE);
			dim3 blocks (gridSizeX, gridSizeY);

			fn.gpu().SetAsCurrent();
			Sonsode::Kernels::Kernel_FullSearch_2D<<<blocks, threads>>>(fn);
			fn.gpu().Synchronize();
		}

		#pragma endregion
	}
}