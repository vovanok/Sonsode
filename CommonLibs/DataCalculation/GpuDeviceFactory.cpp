#include "GpuDeviceFactory.h"

namespace Sonsode {
	std::map<size_t, GpuDevice> GpuDeviceFactory::allowedDevices;

	GpuDevice GpuDeviceFactory::fakeGpuDevice(true);

	std::vector<GpuDevice*> GpuDeviceFactory::GpuDevices() {
		if (allowedDevices.empty()) {
			int countDevices = 0;
			GpuDevice::CheckErr(cudaGetDeviceCount(&countDevices));
		
			cudaDeviceProp props;
			for (size_t devNum = 0; devNum < countDevices; devNum++) {
			GpuDevice::CheckErr(cudaGetDeviceProperties(&props, devNum));
				if (props.major >= 1)
					//allowedDevices.push_back(GpuDevice(devNum));
					allowedDevices[devNum] = GpuDevice(devNum);
			}
		}

		std::vector<GpuDevice*> result;
		for (std::map<size_t, GpuDevice>::iterator iter = allowedDevices.begin(); iter != allowedDevices.end(); iter++) {
			result.push_back(&iter->second);
		}

		return result;
	}

	GpuDevice& GpuDeviceFactory::GetById(size_t id) {
		auto targetPtr = allowedDevices.find(id);
		if (targetPtr != allowedDevices.end())
			return targetPtr->second;
		else
			return fakeGpuDevice;
	}
};