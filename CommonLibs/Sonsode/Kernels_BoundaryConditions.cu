#pragma once

#include "SonsodeCommon.h"
#include "HostData.hpp"
#include "DeviceData.cu"

using Sonsode::HostData2D;
using Sonsode::DeviceData2D;
using Sonsode::BLOCK_SIZE;

namespace Sonsode {
	namespace Kernels {
		template<class FunctorType> void HostKernel_Boundary_2D(FunctorType fn);
		template<class FunctorType> __global__ void Kernel_Boundary_2D(FunctorType fn);

		template<class FunctorType> void HostKernel_Boundary_3D(FunctorType fn);
		template<class FunctorType> __global__ void Kernel_Boundary_3D(FunctorType fn);

		template<class FunctorType> void HostKernel_FullSearch_2D(FunctorType fn);
		template<class FunctorType> __global__ void Kernel_FullSearch_2D(FunctorType fn);
	}
}

namespace Sonsode {
	namespace Kernels {
		template<class FunctorType>
		void HostKernel_Boundary_2D(FunctorType fn) {
			//x boundaries
			for (size_t y = 0; y < fn.dimY(); y++) {
				fn.setValue(0, y, fn.xMinBoundary(y, fn.getValue(0, y), fn.getValue(1, y)));
				fn.setValue(fn.dimX() - 1, y, fn.xMaxBoundary(y, fn.getValue(fn.dimX() - 1, y), fn.getValue(fn.dimX() - 2, y)));
			}
			
			//y boundaries
			for (size_t x = 0; x < fn.dimX(); x++) {
				fn.setValue(x, 0, fn.yMinBoundary(x, fn.getValue(x, 0), fn.getValue(x, 1)));
				fn.setValue(x, fn.dimY() - 1, fn.yMaxBoundary(x, fn.getValue(x, fn.dimY() - 1), fn.getValue(x, fn.dimY() - 2)));
			}
		}

		template<class FunctorType>
		__global__ void Kernel_Boundary_2D(FunctorType fn) {
			size_t coord = blockIdx.x * blockDim.x + threadIdx.x;

			//x boundaries
			if (coord <= fn.dimY() - 1) {
				fn.setValue(0, coord,
					fn.xMinBoundary(coord, fn.getValue(0, coord), fn.getValue(1, coord)));
				fn.setValue(fn.dimX() - 1, coord,
					fn.xMaxBoundary(coord, fn.getValue(fn.dimX() - 1, coord), fn.getValue(fn.dimX() - 2, coord)));
			}

			//y boundaries
			if (coord <= fn.dimX() - 1) {
				fn.setValue(coord, 0,
					fn.yMinBoundary(coord, fn.getValue(coord, 0), fn.getValue(coord, 1)));
				fn.setValue(coord, fn.dimY() - 1,
					fn.yMaxBoundary(coord, fn.getValue(coord, fn.dimY() - 1), fn.getValue(coord, fn.dimY() - 2)));
			}
		}


		template<class FunctorType>
		void HostKernel_Boundary_3D(FunctorType fn) {
			//x boundaries
			for (size_t y = 0; y < fn.dimY(); y++) {
				for (size_t z = 0; z < fn.dimZ(); z++) {
					fn.setValue(0, y, z, fn.xMinBoundary(y, z, fn.getValue(0, y, z), fn.getValue(1, y, z)));
					fn.setValue(fn.dimX() - 1, y, z, fn.xMaxBoundary(y, z, fn.getValue(fn.dimX() - 1, y, z), fn.getValue(fn.dimX() - 2, y, z)));
				}
			}
			
			//y boundaries
			for (size_t x = 0; x < fn.dimX(); x++) {
				for (size_t z = 0; z < fn.dimZ(); z++) {
					fn.setValue(x, 0, z, fn.yMinBoundary(x, z, fn.getValue(x, 0, z), fn.getValue(x, 1, z)));
					fn.setValue(x, fn.dimY() - 1, z, fn.yMaxBoundary(x, z, fn.getValue(x, fn.dimY() - 1, z), fn.getValue(x, fn.dimY() - 2, z)));
				}
			}

			// z boundaries
			for (size_t x = 0; x < fn.dimX(); x++) {
				for (size_t y = 0; y < fn.dimY(); y++) {
					fn.setValue(x, y, 0, fn.yMinBoundary(x, y, fn.getValue(x, y, 0), fn.getValue(x, y, 1)));
					fn.setValue(x, y, fn.dimZ() - 1, fn.yMaxBoundary(x, y, fn.getValue(x, y, fn.dimZ() - 1), fn.getValue(x, y, fn.dimZ() - 2)));
				}
			}
		}

		template<class FunctorType>
		__global__ void Kernel_Boundary_3D(FunctorType fn) {
			size_t coord1 = blockIdx.x * blockDim.x + threadIdx.x;
			size_t coord2 = blockIdx.y * blockDim.y + threadIdx.y;

			//x boundaries
			if (coord1 < fn.dimY() && coord2 < fn.dimZ()) {
				fn.setValue(0, coord1, coord2,
					fn.xMinBoundary(coord1, coord2, fn.getValue(0, coord1, coord2), fn.getValue(1, coord1, coord2)));
				fn.setValue(fn.dimX() - 1, coord1, coord2,
					fn.xMaxBoundary(coord1, coord2, fn.getValue(fn.dimX() - 1, coord1, coord2), fn.getValue(fn.dimX() - 2, coord1, coord2)));
			}

			//y boundaries
			if (coord1 < fn.dimX() && coord2 < fn.dimZ()) {
				fn.setValue(coord1, 0, coord2,
					fn.yMinBoundary(coord1, coord2, fn.getValue(coord1, 0, coord2), fn.getValue(coord1, 1, coord2)));
				fn.setValue(coord1, fn.dimY() - 1, coord2,
					fn.yMaxBoundary(coord1, coord2, fn.getValue(coord1, fn.dimY() - 1, coord2), fn.getValue(coord1, fn.dimY() - 2, coord2)));
			}

			//z boundaries
			if (coord1 < fn.dimX() && coord2 < fn.dimY()) {
				fn.setValue(coord1, coord2, 0,
					fn.zMinBoundary(coord1, coord2, fn.getValue(coord1, coord2, 0), fn.getValue(coord1, coord2, 1)));
				fn.setValue(coord1, coord2, fn.dimZ() - 1,
					fn.zMaxBoundary(coord1, coord2, fn.getValue(coord1, coord2, fn.dimZ() - 1), fn.getValue(coord1, coord2, fn.dimZ() - 2)));
			}
		}


		template<class FunctorType>
		void HostKernel_FullSearch_2D(FunctorType fn) {
			for (size_t x = 0; x < fn.dimX(); x++) {
				for (size_t y = 0; y < fn.dimY(); y++) {
					fn.Action(x, y);
				}
			}
		}

		template<class FunctorType>
		__global__ void Kernel_FullSearch_2D(FunctorType fn) {
			size_t x = blockIdx.x * blockDim.x + threadIdx.x;
			size_t y = blockIdx.y * blockDim.y + threadIdx.y;

			fn.Action(x, y);
		}
	}
}