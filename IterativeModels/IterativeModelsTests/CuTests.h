#pragma once

#include "DeviceData.cu"

namespace CuTests {
	void RunTestKernel1(Sonsode::DeviceData2D<float> data);
	void RunTestKernel2(float *data, size_t dimX, size_t dimY);
}