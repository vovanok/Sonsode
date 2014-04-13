#include "GpuDevice.hpp"

namespace Sonsode {
	GpuDevice::GpuDevice() {
		Initialize(DEFAULT_GPU_ID, false);
	}

	GpuDevice::GpuDevice(size_t id) {
		Initialize(id, false);
		float *test = Malloc<float>(1);
		Free(test);
	}

	GpuDevice::GpuDevice(bool isFake) {
		Initialize(DEFAULT_GPU_ID, isFake);
	}

	void GpuDevice::CheckErr(const cudaError_t &error) {
		if (error != cudaSuccess)
				throw std::string(cudaGetErrorString(error));
	}

	void GpuDevice::Initialize(size_t id, bool isFake) {
		_id = id;
		_isFake = isFake;
	}

	void GpuDevice::Close() const {
		if (isFake())
			return;

		SetAsCurrent();
		CheckErr(cudaDeviceReset());
	}

	bool GpuDevice::operator == (const GpuDevice &compareGpu) const {
		return (id() == compareGpu.id() && isFake() == compareGpu.isFake());
	}

	void GpuDevice::SetAsCurrent() const {
		if (isFake())
			return;

		int curDevNum = -1;
		CheckErr(cudaGetDevice(&curDevNum));

		if (id() != curDevNum)
			CheckErr(cudaSetDevice(id()));
	}

	void GpuDevice::CheckLastErr() const {
		if (isFake())
			return;

		SetAsCurrent();
		CheckErr(cudaGetLastError());
	}

	void GpuDevice::Synchronize() const {
		if (isFake())
			return;

		SetAsCurrent();
		CheckErr(cudaDeviceSynchronize());
	}
};