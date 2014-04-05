#pragma once

#include "Vectors.hpp"
#include "FireSpreadSimpleModelPodTypes.cu"
#include "GpuDevice.hpp"
#include "GpuDeviceFactory.h"

using Sonsode::GpuDevice;
using Sonsode::GpuDeviceFactory;

namespace FireSpreadFunctor
{
	template<class DataKind>
	class Base {
	public:
		__host__ __device__ Base() { }
		Base(FireSpreadConsts consts, DataKind data)
			: _consts(consts), _data(data) { }

		__host__ __device__ size_t dimX() { return _data.dimX(); }
		__host__ __device__ size_t dimY() { return _data.dimY(); }

		GpuDevice& gpu() const {
			throw "GPU property on CPU functor failed";
		}

	protected:
		FireSpreadConsts _consts;
		DataKind _data;

		__host__ __device__ float& t(size_t x, size_t y) { return _data.t(x, y); }
		__host__ __device__ float& roFuel(size_t x, size_t y) { return _data.roFuel(x, y); }
		__host__ __device__ float& t4(size_t x, size_t y) { return _data.t4(x, y); }

		__host__ __device__ float h() { return _consts.H; }
		__host__ __device__ float tau() { return _consts.Tau; }
		__host__ __device__ float humidity() { return _consts.Humidity; }
		__host__ __device__ float windAngle() { return _consts.WindAngle; }
		__host__ __device__ float windSpeed() { return _consts.WindSpeed; }
		__host__ __device__ float m2() { return _consts.M2; }
		__host__ __device__ float danu() { return _consts.Danu; }
		__host__ __device__ float temOnBounds() { return _consts.TemOnBounds; }
		__host__ __device__ int iterFireBeginNum() { return _consts.IterFireBeginNum; }
		__host__ __device__ float qbig() { return _consts.Qbig; }
		__host__ __device__ int mstep() { return _consts.Mstep; }
		__host__ __device__ float tzv() { return _consts.Tzv; }
		__host__ __device__ float temKr() { return _consts.TemKr; }
		__host__ __device__ float qlitl() { return _consts.Qlitl; }
		__host__ __device__ float ks() { return _consts.Ks; }
		__host__ __device__ float enviromentTemperature() { return _consts.EnviromentTemperature; }

		__host__ __device__ float hh() { return _consts.Hh; }
		__host__ __device__ float ap() { return _consts.Ap; }
		__host__ __device__ Vector2D<float> windU() { return _consts.WindU; }
		__host__ __device__ float r0() { return _consts.R0; }
	};

	template<class DataKind>
	class T4 : public Base<DataKind> {
	public:
		__host__ __device__ T4() { }
		T4(FireSpreadConsts consts, DataKind data)
			: Base<DataKind>(consts, data) { }

		__host__ __device__ float getValue(size_t x, size_t y) { return t4(x, y); }
		__host__ __device__ void setValue(size_t x, size_t y, float value) { t4(x, y) = value; }

		__host__ __device__ float Formula(size_t x, size_t y, float s, float l , float r, float u, float d) {
			float temX = (windU().x > 0) ? (r0() * (t(x, y) - t(x-1, y))) : (r0() * (t(x+1, y) - t(x, y)));
			float temY = (windU().y > 0) ? (r0() * (t(x, y) - t(x, y-1))) : (r0() * (t(x, y+1) - t(x, y)));

			float temzX = danu() * (t(x+1, y) - 2.0f * t(x, y) + t(x-1, y)) / hh();
			float temzY = danu() * (t(x, y+1) - 2.0f * t(x, y) + t(x, y-1)) / hh();
			
			return temzX + temzY - windU().x * temX - windU().y * temY;
		}
	};

	template<class DataKind>
	class Gorenie : public Base<DataKind> {
	public:
		__host__ __device__ Gorenie() { }
		Gorenie(FireSpreadConsts consts, DataKind data)
			: Base<DataKind>(consts, data) { }

		__host__ __device__ void Action(size_t x, size_t y) {
			float fgor = qbig() * pow(fabs(t(x, y)), mstep()) * exp(-tzv() / t(x, y));
			if (t(x, y) < temKr())
				fgor = 0.0f;
			roFuel(x, y) = roFuel(x, y) / (1.0f + tau() * fgor);
			t(x, y) = t(x, y) + tau() * qlitl() * roFuel(x, y) * fgor / (1.0f + ks() * humidity());
		}
	};

	template<class DataKind>
	class Temperature : public Base<DataKind> {
	public:
		__host__ __device__ Temperature() { }
		Temperature(FireSpreadConsts consts, DataKind data)
			: Base<DataKind>(consts, data) { }

		__host__ __device__ float getValue(size_t x, size_t y) { return t(x, y); }
		__host__ __device__ void setValue(size_t x, size_t y, float value) { t(x, y) = value; }

		__host__ __device__ bool IsZero(size_t x, size_t y) { return false; }
		__host__ __device__ float AlphaX(size_t x, size_t y) { return -ap(); }
		__host__ __device__ float AlphaY(size_t x, size_t y) { return -ap(); }
		__host__ __device__ float BetaX(size_t x, size_t y) { return 1.0f + 2.0f * ap(); }
		__host__ __device__ float BetaY(size_t x, size_t y) { return 1.0f + 2.0f * ap(); }
		__host__ __device__ float GammaX(size_t x, size_t y) { return -ap(); }
		__host__ __device__ float GammaY(size_t x, size_t y) { return -ap(); }

		__host__ __device__ float QX(size_t x, size_t y) {
			return -ap() * t(x-1, y) + (1.0f + 2.0f * ap()) * t(x, y) 	- ap() * t(x+1, y) + tau() * t4(x, y);
		}

		__host__ __device__ float QY(size_t x, size_t y) {
			return -ap() * t(x, y-1) + (1.0f + 2.0f * ap()) * t(x, y) - ap() * t(x, y+1) + tau() * t4(x, y);
		}
	};

	GpuDevice& Base<FireSpreadDataD>::gpu() const {
		return _data.gpu();
	}
}