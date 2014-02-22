#include "CuTests.h"

__global__ void TestKernel1(Sonsode::DeviceData2D<float> data) {
	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= data.dimX() || y >= data.dimY())
		return;

	data(x, y) = data(x, y) + 1;
}

__global__ void TestKernel2(float *data, size_t dimX, size_t dimY) {
	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= dimX || y >= dimY)
		return;

	data[y * dimX + x] = data[y * dimX + x] + 1;
}

void CuTests::RunTestKernel1(Sonsode::DeviceData2D<float> data) {
	dim3 blockDim(1, 1);
	dim3 threadDim(16, 16);

	TestKernel1<<<blockDim, threadDim>>>(data);
}

void CuTests::RunTestKernel2(float *data, size_t dimX, size_t dimY) {
	dim3 blockDim(1, 1);
	dim3 threadDim(16, 16);

	TestKernel2<<<blockDim, threadDim>>>(data, dimX, dimY);
}

