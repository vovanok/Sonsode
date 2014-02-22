#pragma once

#include <string>
#include <vector>
#include <map>
#include <cuda_runtime.h>
#include "SonsodeCommon.h"

namespace Sonsode {
	const size_t DEFAULT_GPU_ID = 0;

	class GpuDevice {
		friend class GpuDeviceFactory;
		friend class std::map<size_t, GpuDevice>;
	
		size_t _id;
		bool _isFake;

		GpuDevice();
		explicit GpuDevice(size_t id);
		explicit GpuDevice(bool isFake);
		static void CheckErr(const cudaError_t &error);
		void Initialize(size_t id, bool isFake);
	public:
		bool operator == (const GpuDevice &compareGpu) const;
		void SetAsCurrent() const;
		template<class T> T *Malloc(const size_t countItems) const;
		template<class T> void Free(T *&devPtr) const;
		template<class T> void CpyTo(T *dst_d, const T *src_h, size_t countItems) const;
		template<class T> void CpyFrom(T *dst_h, const T *src_d, size_t countItems) const;
		template<class T> void CpyInside(T *dst_d, const T *src_d, size_t countItems) const;
		void CheckLastErr() const;
		void Synchronize() const;
		void Close() const;

		size_t id() const { return _id; }
		bool isFake() const { return _isFake; }
	};

	template<class T> T *GpuDevice::Malloc(const size_t countItems) const {
		if (isFake() || countItems == 0)
			return 0;

		SetAsCurrent();
		T *result = 0;
		CheckErr(cudaMalloc(&result, countItems * sizeof(T)));
		return result;
	}

	template<class T> void GpuDevice::Free(T *&devPtr) const {
		if (isFake()) return;

		SetAsCurrent();
		CheckErr(cudaFree(devPtr));
		devPtr = 0;
	}
	
	template<class T> void GpuDevice::CpyTo(T *dst_d, const T *src_h, size_t countItems) const {
		if (isFake() || countItems == 0)
			return;

		SetAsCurrent();
		CheckErr(cudaMemcpy(dst_d, src_h, countItems * sizeof(T), cudaMemcpyHostToDevice));
	}

	template<class T> void GpuDevice::CpyFrom(T *dst_h, const T *src_d, size_t countItems) const {
		if (isFake() || countItems == 0)
			return;

		SetAsCurrent();
		CheckErr(cudaMemcpy(dst_h, src_d, countItems * sizeof(T), cudaMemcpyDeviceToHost));
	}

	template<class T> void GpuDevice::CpyInside(T *dst_d, const T *src_d, size_t countItems) const {
		if (isFake() || countItems == 0)
			return;

		SetAsCurrent();
		CheckErr(cudaMemcpy(dst_d, src_d, countItems * sizeof(T), cudaMemcpyDeviceToDevice));
	}
};