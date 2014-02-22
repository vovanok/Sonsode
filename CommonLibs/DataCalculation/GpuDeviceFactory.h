#pragma once

#include <vector>
#include <map>
#include "SonsodeCommon.h"
#include "GpuDevice.hpp"

namespace Sonsode {
	class GpuDeviceFactory {
		static std::map<size_t, GpuDevice> allowedDevices;
		static GpuDevice fakeGpuDevice;
	public:
		static std::vector<GpuDevice*> GpuDevices();
		static GpuDevice& GetById(size_t id);
	};
};