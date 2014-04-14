#pragma once

#include "GpuDevice.hpp"
#include "GpuDeviceFactory.h"
#include "DeviceData.cu"

namespace Heat3D {
	namespace Functors {
		using Sonsode::DeviceData3D;
		using Sonsode::GpuDevice;
		using Sonsode::GpuDeviceFactory;

		template<class DataKind>
		class Heat3DFunctor {
		protected:
			float _a, _h, _tau;
			DataKind _data;

		public:
			__host__ __device__ Heat3DFunctor() { }
			Heat3DFunctor(float a, float h, float tau, DataKind data)
				: _a(a), _h(h), _tau(tau), _data(data) { }

			__host__ __device__ float getValue(size_t x, size_t y, size_t z) { return _data(x, y, z); }
			__host__ __device__ void setValue(size_t x, size_t y, size_t z, float value) { _data(x, y, z) = value; }
			__host__ __device__ size_t dimX() { return _data.dimX(); }
			__host__ __device__ size_t dimY() { return _data.dimY(); }
			__host__ __device__ size_t dimZ() { return _data.dimZ(); }

			GpuDevice& gpu() const {
				throw std::exception("GPU property on CPU functor failed");
			}

			__host__ __device__ float Formula(size_t x, size_t y, size_t z, float s, float l, float r, float f, float n, float u, float d) {
				return s + ((pow(_a, 2.0f) * _tau) / pow(_h, 2.0f)) * (l + r + f + n + u + d - 6.0f * s);
			}

			__host__ __device__ bool IsZero(size_t x, size_t y, size_t z) { return true; }
			__host__ __device__ float QX(size_t x, size_t y, size_t z) { return _data(x, y, z); }
			__host__ __device__ float QY(size_t x, size_t y, size_t z) { return QX(x, y, z); }
			__host__ __device__ float QZ(size_t x, size_t y, size_t z) { return QX(x, y, z); }
			__host__ __device__ float AlphaX(size_t x, size_t y, size_t z) { return -(_a * _tau / pow(_h, 2.0f)); }
			__host__ __device__ float AlphaY(size_t x, size_t y, size_t z) { return AlphaX(x, y, z); }
			__host__ __device__ float AlphaZ(size_t x, size_t y, size_t z) { return AlphaX(x, y, z); }
			__host__ __device__ float BetaX(size_t x, size_t y, size_t z) { return 1.0f + 2.0f * (_a * _tau / pow(_h, 2.0f)); }
			__host__ __device__ float BetaY(size_t x, size_t y, size_t z) { return BetaX(x, y, z); }
			__host__ __device__ float BetaZ(size_t x, size_t y, size_t z) { return BetaX(x, y, z); }
			__host__ __device__ float GammaX(size_t x, size_t y, size_t z) { return -(_a * _tau / pow(_h, 2.0f)); }
			__host__ __device__ float GammaY(size_t x, size_t y, size_t z) { return GammaX(x, y, z); }
			__host__ __device__ float GammaZ(size_t x, size_t y, size_t z) { return GammaY(x, y, z); }
		};

		GpuDevice& Heat3DFunctor<DeviceData3D<float>>::gpu() const {
			return _data.gpu();
		}
	}
}