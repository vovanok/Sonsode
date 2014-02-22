#pragma once

#include <iostream>
#include "GpuDevice.hpp"
#include "DeviceData.cu"
#include "HostData.hpp"
#include "CuTests.h"

namespace Tests {
	void GpuDevice();
	void DeviceData();
	void GpuDevice2();
}