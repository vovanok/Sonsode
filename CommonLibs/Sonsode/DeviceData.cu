#pragma once

#include "SonsodeCommon.h"
#include "HostData.hpp"
#include "GpuDevice.hpp"
#include "GpuDeviceFactory.h"
#include "SonsodeException.h"

namespace Sonsode {
	template<class T>
	class DeviceData1D {
		size_t _dimX;

		size_t gpuId;
		T *_data;

	public:
		__host__ __device__ DeviceData1D() { }
		DeviceData1D(const GpuDevice &gpuDevice, size_t dimX);
		DeviceData1D(const GpuDevice &gpuDevice, const HostData1D<T>& hostData);

		GpuDevice& gpu() const { return GpuDeviceFactory::GetById(gpuId); }
		void Erase() { gpu().Free(_data); }
		size_t dimX() const { return _dimX; }
		T* data() const { return _data; }
		__device__ T& at(size_t x) const { return this->operator()(x); }
		__device__ T& operator()(size_t x) const { return _data[(x <= _dimX - 1) * (x)]; }

		void TakeFrom(const HostData1D<T> &data_h);
		void PutTo(HostData1D<T> &data_h) const;
	};

	template<class T>
	class DeviceData2D {
		size_t _dimX;
		size_t _dimY;

		size_t gpuId;
		T *_data;

	public:
		__host__ __device__ DeviceData2D() { }
		DeviceData2D(const GpuDevice &gpuDevice, size_t dimX, size_t dimY);
		DeviceData2D(const GpuDevice &gpuDevice, const HostData2D<T>& hostData);

		__host__ __device__ size_t dimX() const { return _dimX; }
		__host__ __device__ size_t dimY() const { return _dimY; }
		T *data() const { return _data; }
		GpuDevice& gpu() const { return GpuDeviceFactory::GetById(gpuId); }
		void Erase() { gpu().Free(_data); }
		__device__ T &at(size_t x, size_t y) const { return this->operator()(x, y); 	}
		__device__ T &operator()(size_t x, size_t y) const {
			return _data[(x <= _dimX - 1 && y <= _dimY - 1) * (y * _dimX + x)];
		}

		void TakeFrom(const HostData2D<T> &data_h);
		void PutTo(HostData2D<T> &data_h) const;
	};

	template<class T>
	class DeviceData3D {
		size_t _dimX;
		size_t _dimY;
		size_t _dimZ;

		size_t gpuId;
		T *_data;

	public:
		__host__ __device__ DeviceData3D() { }
		DeviceData3D(const GpuDevice &gpuDevice, size_t dimX, size_t dimY, size_t dimZ);
		DeviceData3D(const GpuDevice &gpuDevice, const HostData3D<T>& hostData);

		GpuDevice& gpu() const { return GpuDeviceFactory::GetById(gpuId); }
		void Erase() { gpu().Free(_data); }
		__host__ __device__ size_t dimX() const { return _dimX; }
		__host__ __device__ size_t dimY() const { return _dimY; }
		__host__ __device__ size_t dimZ() const { return _dimZ; }
		T* data() const { return _data; }
		__device__ T& at(size_t x, size_t y, size_t z) const { return this->operator()(x, y, z); }
		__device__ T& operator()(size_t x, size_t y, size_t z) const { 
			return _data[(x <= _dimX - 1 && y <= _dimY - 1 && z <= _dimZ - 1) * (z * _dimX * _dimY + y * _dimX + x)];
		}

		void TakeFrom(const HostData3D<T> &data_h);
		void PutTo(HostData3D<T> &data_h) const;
	};

	template<class T>
	class DeviceData4D {
		size_t _dimX;
		size_t _dimY;
		size_t _dimZ;
		size_t _dimW;

		size_t gpuId;
		T *_data;

	public:
		__host__ __device__ DeviceData4D() { }
		DeviceData4D(const GpuDevice &gpuDevice, size_t dimX, size_t dimY, size_t dimZ, size_t dimW);
		DeviceData4D(const GpuDevice &gpuDevice, const HostData4D<T>& hostData);

		GpuDevice& gpu() const {  return GpuDeviceFactory::GetById(gpuId); }
		void Erase() { gpu().Free(_data); }
		__host__ __device__ size_t dimX() const { return _dimX; }
		__host__ __device__ size_t dimY() const { return _dimY; }
		__host__ __device__ size_t dimZ() const { return _dimZ; }
		__host__ __device__ size_t dimW() const { return _dimW; }
		T* data() const { return _data; }
		__device__ T& at(size_t x, size_t y, size_t z, size_t w) const { return this->operator()(x, y, z, w); }
		__device__ T& operator()(size_t x, size_t y, size_t z, size_t w) const { 
			return _data[(x < _dimX && y < _dimY && z < _dimZ && w < _dimW) * (w * _dimX * _dimY * _dimZ + z * _dimX * _dimY + y * _dimX + x)];
		}

		void TakeFrom(const HostData4D<T> &data_h);
		void PutTo(HostData4D<T> &data_h) const;
	};

	#pragma region Implementation

	#pragma region DeviceData1D
	template<class T>
	DeviceData1D<T>::DeviceData1D(const GpuDevice &gpuDevice, size_t dimX) {
		gpuId = gpuDevice.id();
		_dimX = dimX;
		_data = gpu().Malloc<T>(_dimX);
	}

	template<class T>
	DeviceData1D<T>::DeviceData1D(const GpuDevice &gpuDevice, const HostData1D<T>& hostData) {
		gpuId = gpuDevice.id();
		_dimX = hostData.dimX();
		_data = gpu().Malloc<T>(_dimX);
		TakeFrom(hostData);
	}

	template<class T>
	void DeviceData1D<T>::TakeFrom(const HostData1D<T> &data_h) {
		if (dimX() != data_h.dimX())
			throw SonsodeException("Different sizes of data");

		gpu().CpyTo(data(), data_h.data(), dimX());
	}

	template<class T>
	void DeviceData1D<T>::PutTo(HostData1D<T> &data_h) const {
		if (dimX() != data_h.dimX())
			throw SonsodeException("Different sizes of data");

		gpu().CpyFrom(data_h.data(), data(), dimX());
	}
	#pragma endregion

	#pragma region DeviceData2D
	template<class T>
	DeviceData2D<T>::DeviceData2D(const GpuDevice &gpuDevice, size_t dimX, size_t dimY) {
		gpuId = gpuDevice.id();
		_dimX = dimX;
		_dimY = dimY;
		_data = gpu().Malloc<T>(_dimX * _dimY);
	}

	template<class T>
	DeviceData2D<T>::DeviceData2D(const GpuDevice &gpuDevice, const HostData2D<T>& hostData) {
		gpuId = gpuDevice.id();
		_dimX = hostData.dimX();
		_dimY = hostData.dimY();
		_data = gpu().Malloc<T>(_dimX * _dimY);
		TakeFrom(hostData);
	}

	template<class T>
	void DeviceData2D<T>::TakeFrom(const HostData2D<T> &data_h) {
		if (dimX() != data_h.dimX() || dimY() != data_h.dimY())
			throw SonsodeException("Different sizes of data");

		gpu().CpyTo(data(), data_h.data(), dimX() * dimY());
	}

	template<class T>
	void DeviceData2D<T>::PutTo(HostData2D<T> &data_h) const {
		if (dimX() != data_h.dimX() || dimY() != data_h.dimY())
			throw SonsodeException("Different sizes of data");

		gpu().CpyFrom(data_h.data(), data(), dimX() * dimY());
	}
	#pragma endregion

	#pragma region DeviceData3D
	template<class T>
	DeviceData3D<T>::DeviceData3D(const GpuDevice &gpuDevice, size_t dimX, size_t dimY, size_t dimZ) {
		gpuId = gpuDevice.id();
		_dimX = dimX;
		_dimY = dimY;
		_dimZ = dimZ;
		_data = gpu().Malloc<T>(_dimX * _dimY * _dimZ);
	}

	template<class T>
	DeviceData3D<T>::DeviceData3D(const GpuDevice &gpuDevice, const HostData3D<T>& hostData) {
		gpuId = gpuDevice.id();
		_dimX = hostData.dimX();
		_dimY = hostData.dimY();
		_dimZ = hostData.dimZ();
		_data = gpu().Malloc<T>(_dimX * _dimY * _dimZ);
		TakeFrom(hostData);
	}

	template<class T>
	void DeviceData3D<T>::TakeFrom(const HostData3D<T>& data_h) {
		if (dimX() != data_h.dimX() || dimY() != data_h.dimY() || dimZ() != data_h.dimZ())
			throw SonsodeException("Different sizes of data");

		gpu().CpyTo(data(), data_h.data(), dimX() * dimY() * dimZ());
	}

	template<class T>
	void DeviceData3D<T>::PutTo(HostData3D<T> &data_h) const {
		if (dimX() != data_h.dimX() || dimY() != data_h.dimY() || dimZ() != data_h.dimZ())
			throw SonsodeException("Different sizes of data");

		gpu().CpyFrom(data_h.data(), data(), dimX() * dimY() * dimZ());
	}
	#pragma endregion

	#pragma region DeviceData4D
	template<class T>
	DeviceData4D<T>::DeviceData4D(const GpuDevice &gpuDevice, size_t dimX, size_t dimY, size_t dimZ, size_t dimW) {
		gpuId = gpuDevice.id();
		_dimX = dimX;
		_dimY = dimY;
		_dimZ = dimZ;
		_dimW = dimW;
		_data = gpu().Malloc<T>(_dimX * _dimY * _dimZ * _dimW);
	}

	template<class T>
	DeviceData4D<T>::DeviceData4D(const GpuDevice &gpuDevice, const HostData4D<T>& hostData) {
		gpuId = gpuDevice.id();
		_dimX = hostData.dimX();
		_dimY = hostData.dimY();
		_dimZ = hostData.dimZ();
		_dimW = hostData.dimW();
		_data = gpu().Malloc<T>(_dimX * _dimY * _dimZ * _dimW);
		TakeFrom(hostData);
	}

	template<class T>
	void DeviceData4D<T>::TakeFrom(const HostData4D<T>& data_h) {
		if (dimX() != data_h.dimX() || dimY() != data_h.dimY() || dimZ() != data_h.dimZ() || dimW() != data_h.dimW())
			throw SonsodeException("Different sizes of data");

		gpu().CpyTo(data(), data_h.data(), dimX() * dimY() * dimZ() * dimW());
	}

	template<class T>
	void DeviceData4D<T>::PutTo(HostData4D<T> &data_h) const {
		if (dimX() != data_h.dimX() || dimY() != data_h.dimY() || dimZ() != data_h.dimZ() || dimW() != data_h.dimW())
			throw SonsodeException("Different sizes of data");

		gpu().CpyFrom(data_h.data(), data(), dimX() * dimY() * dimZ() * dimW());
	}
	#pragma endregion

	#pragma endregion
};