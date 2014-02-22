#pragma once

#include "SonsodeCommon.h"
#include "HostData.hpp"
#include "DeviceData.cu"

using Sonsode::HostData2D;
using Sonsode::DeviceData2D;
using Sonsode::BLOCK_SIZE;

namespace {
	__device__ size_t FieldIndexByThread(size_t blockDim, size_t blockIdx,
																			 size_t threadIdx, size_t overlayValue) {
		return blockIdx * (blockDim - (2 * overlayValue)) + threadIdx;
	}
}

namespace Sonsode {
	namespace Kernels {
		//Явная разностная схема Гаусса - Зельделя на CPU 2D
		template<class FunctorType> void HostKernel_ExplicitGaussSeidel_2D(FunctorType fn);

		//Явная разностная схема Гаусса - Зельделя на GPU 2D (прямой подход)
		template<class FunctorType> __global__ void Kernel_ExplicitGaussSeidel_2D_Direct(FunctorType fn);

		//Явная разностная схема Гаусса - Зельделя на GPU 2D (шахматный подход)
		template<class FunctorType> __global__ void Kernel_ExplicitGaussSeidel_2D_Chess(FunctorType fn);

		//Явная разностная схема Гаусса - Зельделя на GPU 2D (бесконфликтный подход)
		template<class FunctorType> __global__ void Kernel_ExplicitGaussSeidel_2D_OutConf(FunctorType fn);

		//Явная разностная схема Гаусса - Зельделя на GPU 2D (прямой подход нахлестом блоков друг на друга)
		template<class FunctorType> __global__ void Kernel_ExplicitGaussSeidel_2D_DirectOverlay(FunctorType fn);


		//Явная разностная схема Гаусса - Зельделя на CPU 3D
		template<class FunctorType> void HostKernel_ExplicitGaussSeidel_3D(FunctorType fn);

		//Явная разностная схема Гаусса - Зельделя на GPU 3D (прямой подход)
		template<class FunctorType> __global__ void Kernel_ExplicitGaussSeidel_3D_Direct(FunctorType fn);
	}
}


namespace Sonsode {
	namespace Kernels {
		template<class FunctorType>
		void HostKernel_ExplicitGaussSeidel_2D(FunctorType fn) {
			for (size_t x = 1; x <= fn.dimX() - 2; x++)
				for (size_t y = 1; y <= fn.dimY() - 2; y++)
					fn.setValue(x, y, fn.Formula(x, y, fn.getValue(x, y), fn.getValue(x-1, y), fn.getValue(x+1, y), fn.getValue(x, y-1), fn.getValue(x, y+1)));	
		}

		template<class FunctorType>
		__global__ void Kernel_ExplicitGaussSeidel_2D_Direct(FunctorType fn) {
			//высчитываю индексы точки в поле за которую я ответственнен
			size_t x = blockIdx.x * blockDim.x + threadIdx.x;
			size_t y = blockIdx.y * blockDim.y + threadIdx.y;

			//выделяем для блока разделяемую память размером блока + место для граничных точек
			__shared__ float subPoly [BLOCK_SIZE + 2][BLOCK_SIZE + 2];
	
			if (x > fn.dimX() - 1 || y > fn.dimY() - 1)
				return;

			//перемещаю свою точку в разделяемую память
			subPoly[threadIdx.x + 1][threadIdx.y + 1] = fn.getValue(x, y);

			if (x == 0 || x == fn.dimX() - 1 || y == 0 || y == fn.dimY() - 1)
				return;

			//если я на границе блока, перемещаю граничные точки в разделяемую память
			if (threadIdx.x == 0) //перенос данных в левую границу
				subPoly[threadIdx.x][threadIdx.y + 1] = fn.getValue(x - 1, y);

			if (threadIdx.y == 0) //перенос данных в нижнюю границу
				subPoly[threadIdx.x + 1][threadIdx.y] = fn.getValue(x, y - 1);

			if (threadIdx.x == blockDim.x - 1) //перенос данных в правую границу
				subPoly[threadIdx.x + 2][threadIdx.y + 1] = fn.getValue(x + 1, y);

			if (threadIdx.y == blockDim.y - 1) //перенос данных с верхнюю границу
				subPoly[threadIdx.x + 1][threadIdx.y + 2] = fn.getValue(x, y + 1);

			__syncthreads();

			for (size_t subIter = 0; subIter < COUNT_LOCAL_ITERATIONS; subIter++)	{
				//вычисляю свою точку по формуле
				subPoly[threadIdx.x + 1][threadIdx.y + 1] =
					fn.Formula(	x, y,
											subPoly[threadIdx.x + 1][threadIdx.y + 1],
											subPoly[threadIdx.x][threadIdx.y + 1],
											subPoly[threadIdx.x + 2][threadIdx.y + 1],
											subPoly[threadIdx.x + 1][threadIdx.y],
											subPoly[threadIdx.x + 1][threadIdx.y + 2]);

				__syncthreads();
			}

			//записываю результат в глобальную память
			fn.setValue(x, y, subPoly[threadIdx.x + 1][threadIdx.y + 1]);
		}

		template<class FunctorType>
		__global__ void Kernel_ExplicitGaussSeidel_2D_Chess(FunctorType fn) {
			//высчитываю индексы точки в поле за которую я ответственнен
			size_t x = blockIdx.x * blockDim.x + threadIdx.x;
			size_t y = blockIdx.y * blockDim.y + threadIdx.y;

			//выделяем для блока разделяемую память размером блока + место для граничных точек
			__shared__ float subPoly [BLOCK_SIZE + 2][BLOCK_SIZE + 2];
	
			//если мне не досталось точки, то завершаю работу
			if (x > fn.dimX() - 1 || y > fn.dimY() - 1)
				return;

			//перемещаю свою точку в разделяемую память
			subPoly[threadIdx.x + 1][threadIdx.y + 1] = fn.getValue(x, y);

			//я на границе, то завершаю работу
			if (x == 0 || x == fn.dimX() - 1 || y == 0 || y == fn.dimY() - 1)
				return;

			//если я на границе блока, перемещаю граничные точки в разделяемую память
			if (threadIdx.x == 0) //перенос данных в левую границу
				subPoly[threadIdx.x][threadIdx.y + 1] = fn.getValue(x - 1, y);

			if (threadIdx.y == 0) //перенос данных в нижнюю границу
				subPoly[threadIdx.x + 1][threadIdx.y] = fn.getValue(x, y - 1);

			if (threadIdx.x == blockDim.x - 1) //перенос данных в правую границу
				subPoly[threadIdx.x + 2][threadIdx.y + 1] = fn.getValue(x + 1, y);

			if (threadIdx.y == blockDim.y - 1) //перенос данных с верхнюю границу
				subPoly[threadIdx.x + 1][threadIdx.y + 2] = fn.getValue(x, y + 1);

			__syncthreads();

			for (size_t subIter = 0; subIter < COUNT_LOCAL_ITERATIONS; subIter++)	{
				//этап 1: белые клетки
				if ((x + y) % 2 == 0) {
					subPoly[threadIdx.x + 1][threadIdx.y + 1]	=
						fn.Formula(	x, y,
												subPoly[threadIdx.x + 1][threadIdx.y + 1],
												subPoly[threadIdx.x][threadIdx.y + 1],
												subPoly[threadIdx.x + 2][threadIdx.y + 1],
												subPoly[threadIdx.x + 1][threadIdx.y],
												subPoly[threadIdx.x + 1][threadIdx.y + 2]);
				}
				__syncthreads();

				//этап 2: черные клетки
				if ((x + y) % 2 != 0) {
					subPoly[threadIdx.x + 1][threadIdx.y + 1] =
						fn.Formula(	x, y,
												subPoly[threadIdx.x + 1][threadIdx.y + 1],
												subPoly[threadIdx.x][threadIdx.y + 1],
												subPoly[threadIdx.x + 2][threadIdx.y + 1],
												subPoly[threadIdx.x + 1][threadIdx.y],
												subPoly[threadIdx.x + 1][threadIdx.y + 2]);
				}
				__syncthreads();
			}	
			//записываю результат в глобальную память
			fn.setValue(x, y, subPoly[threadIdx.x + 1][threadIdx.y + 1]);
		}

		template<class FunctorType>
		__global__ void Kernel_ExplicitGaussSeidel_2D_OutConf(FunctorType fn) {
			//высчитываю индексы точки в поле за которую я ответственнен
			size_t x = blockIdx.x * blockDim.x + threadIdx.x;
			size_t y = blockIdx.y * blockDim.y + threadIdx.y;

			//выделяем для блока разделяемую память размером блока + место для граничных точек
			__shared__ float subPoly [BLOCK_SIZE + 2][BLOCK_SIZE + 2];
	
			//если мне не досталось точки, то завершаю работу
			if (x > fn.dimX() - 1 || y > fn.dimY() - 1)
				return;

			//перемещаю свою точку в разделяемую память
			subPoly[threadIdx.x + 1][threadIdx.y + 1] = fn.getValue(x, y);

			//я на границе, то завершаю работу
			if (x == 0 || x == fn.dimX() - 1 || y == 0 || y == fn.dimY() - 1)
				return;
	
			//если я на границе блока, перемещаю граничные точки в разделяемую память
			if (threadIdx.x == 0) //перенос данных в левую границу
				subPoly[threadIdx.x][threadIdx.y + 1] = fn.getValue(x-1, y);
			if (threadIdx.y == 0) //перенос данных в нижнюю границу
				subPoly[threadIdx.x + 1][threadIdx.y] = fn.getValue(x, y-1);
			if (threadIdx.x == blockDim.x - 1) //перенос данных в правую границу
				subPoly[threadIdx.x + 2][threadIdx.y + 1] = fn.getValue(x+1, y);
			if (threadIdx.y == blockDim.y - 1) //перенос данных с верхнюю границу
				subPoly[threadIdx.x + 1][threadIdx.y + 2] = fn.getValue(x, y+1);
			__syncthreads();

			for (size_t subIter = 0; subIter < COUNT_LOCAL_ITERATIONS; subIter++)	{
				//этап 1
				if ((threadIdx.x % 3 == 1 && threadIdx.y % 6 == 1) ||
						(threadIdx.x % 3 == 2 && threadIdx.y % 6 == 3) ||
						(threadIdx.x % 3 == 0 && threadIdx.y % 6 == 5)) {
					subPoly[threadIdx.x + 1][threadIdx.y + 1]	=
						fn.Formula(	x, y,
												subPoly[threadIdx.x + 1][threadIdx.y + 1],
												subPoly[threadIdx.x][threadIdx.y + 1],
												subPoly[threadIdx.x + 2][threadIdx.y + 1],
												subPoly[threadIdx.x + 1][threadIdx.y],
												subPoly[threadIdx.x + 1][threadIdx.y + 2]);
				}
				__syncthreads();

				//этап 2
				if ((threadIdx.x % 3 == 2 && threadIdx.y % 6 == 1) ||
						(threadIdx.x % 3 == 0 && threadIdx.y % 6 == 3) ||
						(threadIdx.x % 3 == 1 && threadIdx.y % 6 == 5)) {
					subPoly[threadIdx.x + 1][threadIdx.y + 1]	=
						fn.Formula(	x, y,
												subPoly[threadIdx.x + 1][threadIdx.y + 1],
												subPoly[threadIdx.x][threadIdx.y + 1],
												subPoly[threadIdx.x + 2][threadIdx.y + 1],
												subPoly[threadIdx.x + 1][threadIdx.y],
												subPoly[threadIdx.x + 1][threadIdx.y + 2]);
				}
				__syncthreads();
				//этап 3
				if ((threadIdx.x % 3 == 0 && threadIdx.y % 6 == 1) ||
						(threadIdx.x % 3 == 1 && threadIdx.y % 6 == 3) ||
						(threadIdx.x % 3 == 2 && threadIdx.y % 6 == 5)) {
					subPoly[threadIdx.x + 1][threadIdx.y + 1]	=
						fn.Formula(	x, y,
												subPoly[threadIdx.x + 1][threadIdx.y + 1],
												subPoly[threadIdx.x][threadIdx.y + 1],
												subPoly[threadIdx.x + 2][threadIdx.y + 1],
												subPoly[threadIdx.x + 1][threadIdx.y],
												subPoly[threadIdx.x + 1][threadIdx.y + 2]);
				}
				__syncthreads();
				//этап 4
				if ((threadIdx.x % 3 == 1 && threadIdx.y % 6 == 2) ||
						(threadIdx.x % 3 == 2 && threadIdx.y % 6 == 4) ||
						(threadIdx.x % 3 == 0 && threadIdx.y % 6 == 0)) {
					subPoly[threadIdx.x + 1][threadIdx.y + 1]	=
						fn.Formula(	x, y,
												subPoly[threadIdx.x + 1][threadIdx.y + 1],
												subPoly[threadIdx.x][threadIdx.y + 1],
												subPoly[threadIdx.x + 2][threadIdx.y + 1],
												subPoly[threadIdx.x + 1][threadIdx.y],
												subPoly[threadIdx.x + 1][threadIdx.y + 2]);
				}
				__syncthreads();
				//этап 5
				if ((threadIdx.x % 3 == 2 && threadIdx.y % 6 == 2) ||
						(threadIdx.x % 3 == 0 && threadIdx.y % 6 == 4) ||
						(threadIdx.x % 3 == 1 && threadIdx.y % 6 == 0)) {
					subPoly[threadIdx.x + 1][threadIdx.y + 1]	=
						fn.Formula(	x, y,
												subPoly[threadIdx.x + 1][threadIdx.y + 1],
												subPoly[threadIdx.x][threadIdx.y + 1],
												subPoly[threadIdx.x + 2][threadIdx.y + 1],
												subPoly[threadIdx.x + 1][threadIdx.y],
												subPoly[threadIdx.x + 1][threadIdx.y + 2]);
				}
				__syncthreads();
				//этап 6
				if ((threadIdx.x % 3 == 0 && threadIdx.y % 6 == 2) ||
						(threadIdx.x % 3 == 1 && threadIdx.y % 6 == 4) ||
						(threadIdx.x % 3 == 2 && threadIdx.y % 6 == 0))	{
					subPoly[threadIdx.x + 1][threadIdx.y + 1]	=
						fn.Formula(	x, y,
												subPoly[threadIdx.x + 1][threadIdx.y + 1],
												subPoly[threadIdx.x][threadIdx.y + 1],
												subPoly[threadIdx.x + 2][threadIdx.y + 1],
												subPoly[threadIdx.x + 1][threadIdx.y],
												subPoly[threadIdx.x + 1][threadIdx.y + 2]);
				}
				__syncthreads();
			}

			//записываю результат в глобальную память
			fn.setValue(x, y, subPoly[threadIdx.x + 1][threadIdx.y + 1]);
		}

		template<class FunctorType>
		__global__ void Kernel_ExplicitGaussSeidel_2D_DirectOverlay(FunctorType fn, size_t overlayValue) {
			size_t x = FieldIndexByThread(blockDim.x, blockIdx.x, threadIdx.x, overlayValue);
			size_t y = FieldIndexByThread(blockDim.y, blockIdx.y, threadIdx.y, overlayValue);

			__shared__ float subPoly [BLOCK_SIZE][BLOCK_SIZE];
	
			if (x > fn.dimX() - 1 || y > fn.dimY() - 1)
				return;

			subPoly[threadIdx.x][threadIdx.y] = fn.getValue(x, y);
			__syncthreads();

			if (threadIdx.x == 0 || threadIdx.x == blockDim.x - 1 ||
					threadIdx.y == 0 || threadIdx.y == blockDim.y - 1 ||
					x == fn.dimX() - 1 || y == fn.dimY() - 1)
				return;

			for (size_t subIter = 0; subIter < COUNT_LOCAL_ITERATIONS; subIter++)	{
				subPoly[threadIdx.x][threadIdx.y] =
					fn.Formula(	x, y,
											subPoly[threadIdx.x][threadIdx.y],
											subPoly[threadIdx.x - 1][threadIdx.y],
											subPoly[threadIdx.x + 1][threadIdx.y],
											subPoly[threadIdx.x][threadIdx.y - 1],
											subPoly[threadIdx.x][threadIdx.y + 1]);

				__syncthreads();
			}

			//записываю результат в глобальную память
			fn.setValue(x, y, subPoly[threadIdx.x][threadIdx.y]);
		}


		template<class FunctorType>
		void HostKernel_ExplicitGaussSeidel_3D(FunctorType fn) {
			for (size_t x = 1; x <= fn.dimX() - 2; x++)
				for (size_t y = 1; y <= fn.dimY() - 2; y++)
					for (size_t z = 1; z <= fn.dimZ() - 2; z++)
						fn.setValue(x, y, z, fn.Formula(x, y, z,
																						fn.getValue(x, y, z),
																						fn.getValue(x-1, y, z), fn.getValue(x+1, y, z),
																						fn.getValue(x, y-1, z), fn.getValue(x, y+1, z),
																						fn.getValue(x, y, z-1), fn.getValue(x, y, z+1)));
		}

		template<class FunctorType>
		__global__ void Kernel_ExplicitGaussSeidel_3D_Direct(
			FunctorType fn, size_t gridDimX, size_t gridDimY, size_t gridDimZ) {

			//называю свои индексы блока и потока короткими именами
			int bx = (blockIdx.x % (gridDimX * gridDimY)) % gridDimX;
			int by = (blockIdx.x % (gridDimX * gridDimY)) / gridDimX;
			int bz = blockIdx.x / (gridDimX * gridDimY);

			//высчитываю индексы точки в поле за которую я ответственнен
			int x = bx * blockDim.x + threadIdx.x;
			int y = by * blockDim.y + threadIdx.y;
			int z = bz * blockDim.z + threadIdx.z;

			//выделяем для блока разделяемую память размером блока + место для граничных точек
			__shared__ float subPoly [BLOCK_SIZE + 2][BLOCK_SIZE + 2][BLOCK_SIZE + 2];

			if (!(x >= 0 && x <= fn.dimX() - 1 && y >= 0 && y <= fn.dimY() - 1 && z >= 0 && z <= fn.dimZ() - 1))
				return;
	
			//перемещаю свою точку в разделяемую память
			subPoly[threadIdx.x + 1][threadIdx.y + 1][threadIdx.z + 1] = fn.getValue(x, y, z);

			//если я на границе, перемещаю граничные точки в разделяемую память
			if (bx != 0 && threadIdx.x == 0) //Левая граница
				subPoly[threadIdx.x][threadIdx.y + 1][threadIdx.z + 1] = fn.getValue(x-1, y, z);
			if (by != 0 && threadIdx.y == 0) //Задняя граница
				subPoly[threadIdx.x + 1][threadIdx.y][threadIdx.z + 1] = fn.getValue(x, y-1, z);
			if (bz != 0 && threadIdx.z == 0) //Верхняя граница
				subPoly[threadIdx.x + 1][threadIdx.y + 1][threadIdx.z] = fn.getValue(x, y, z-1);

			if (bx != gridDimX - 1 && threadIdx.x == blockDim.x - 1) //Правая граница
				subPoly[threadIdx.x + 2][threadIdx.y + 1][threadIdx.z + 1] = fn.getValue(x+1, y, z);
			if (by != gridDimY - 1 && threadIdx.y == blockDim.y - 1) //Передняя граница
				subPoly[threadIdx.x + 1][threadIdx.y + 2][threadIdx.z + 1] = fn.getValue(x, y+1, z);
			if (bz != gridDimZ - 1 && threadIdx.z == blockDim.z - 1) //Нижняя граница
				subPoly[threadIdx.x + 1][threadIdx.y + 1][threadIdx.z + 2] = fn.getValue(x, y, z+1);

			__syncthreads();

			//если моя точка не граничная
			if (x != 0 && x != fn.dimX() - 1 && y != 0 && y != fn.dimY() - 1 && z != 0 && z != fn.dimZ() - 1) 	{
				//выполняю countIterations подитераций над своей точкой
				for (int subIter = 0; subIter < COUNT_LOCAL_ITERATIONS; subIter++) {
						subPoly[threadIdx.x + 1][threadIdx.y + 1][threadIdx.z + 1] = 
							fn.Formula(	x, y, z,
													subPoly[threadIdx.x + 1][threadIdx.y + 1][threadIdx.z + 1],
													subPoly[threadIdx.x][threadIdx.y + 1][threadIdx.z + 1],
													subPoly[threadIdx.x + 2][threadIdx.y + 1][threadIdx.z + 1],
													subPoly[threadIdx.x + 1][threadIdx.y][threadIdx.z + 1],
													subPoly[threadIdx.x + 1][threadIdx.y + 2][threadIdx.z + 1],
													subPoly[threadIdx.x + 1][threadIdx.y + 1][threadIdx.z],
													subPoly[threadIdx.x + 1][threadIdx.y + 1][threadIdx.z + 2]);

					__syncthreads();
				}
			}

			//записываю свой результат в глобальную память
			fn.setValue(x, y, z, subPoly[threadIdx.x + 1][threadIdx.y + 1][threadIdx.z + 1]);
		}
	}
}