#pragma once

#include "SonsodeCommon.h"
#include "DeviceData.cu"
#include "GpuDevice.hpp"
#include "GpuDeviceFactory.h"
#include "SweepFactors.cu"

namespace Heat2D {
	namespace Functors {
		using Sonsode::DeviceData2D;
		using Sonsode::GpuDevice;
		using Sonsode::GpuDeviceFactory;

		template<class DataKind>
		class Heat2DFunctor {
		protected:
			float _a, _h, _tau;
			DataKind _data;

		public:
			__host__ __device__ Heat2DFunctor() { }
			Heat2DFunctor(float a, float h, float tau, DataKind data)
				: _a(a), _h(h), _tau(tau), _data(data) { }

			__host__ __device__ float getValue(size_t x, size_t y) { return _data(x, y); }
			__host__ __device__ void setValue(size_t x, size_t y, float value) { _data(x, y) = value; }
			__host__ __device__ size_t dimX() { return _data.dimX(); }
			__host__ __device__ size_t dimY() { return _data.dimY(); }
	
			GpuDevice& gpu() const {
				throw std::exception("GPU property on CPU functor failed");
			}

			__host__ __device__ float Formula(size_t x, size_t y, float s, float l , float r, float u, float d) {
				return s + ((pow(_a, 2.0f) * _tau) / pow(_h, 2.0f)) * (l + r + u + d - 4.0f * s);
			}

			__host__ __device__ bool IsZero(size_t x, size_t y) { return true; }
			__host__ __device__ float QX(size_t x, size_t y) { return _data(x, y); }
			__host__ __device__ float QY(size_t x, size_t y) { return QX(x, y); }
			__host__ __device__ float AlphaX(size_t x, size_t y) { return -(_a * _tau / pow(_h, 2.0f)); }
			__host__ __device__ float AlphaY(size_t x, size_t y) { return AlphaX(x, y); }
			__host__ __device__ float BetaX(size_t x, size_t y) { return 1.0f + 2.0f * (_a * _tau / pow(_h, 2.0f)); }
			__host__ __device__ float BetaY(size_t x, size_t y) { return BetaX(x, y); }
			__host__ __device__ float GammaX(size_t x, size_t y) { return -(_a * _tau / pow(_h, 2.0f)); }
			__host__ __device__ float GammaY(size_t x, size_t y) { return GammaX(x, y); }
		};

		GpuDevice& Heat2DFunctor<DeviceData2D<float>>::gpu() const {
			return _data.gpu();
		}
	}
}