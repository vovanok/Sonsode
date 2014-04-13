#pragma once

#include "AirFlowPodTypes.cu"
#include "GpuDevice.hpp"
#include "GpuDeviceFactory.h"

namespace AirFlow {
	namespace Functors {
		using Sonsode::GpuDevice;
		using Sonsode::GpuDeviceFactory;

		template<class DataKind>
		class Base {
		public:
			__device__ __host__ Base() { }
			Base(AirFlowConsts consts, DataKind data)
			 : _consts(consts), _data(data) { }

			__device__ __host__ size_t dimX() { return _data.dimX(); }
			__device__ __host__ size_t dimY() { return _data.dimY(); }
			__device__ __host__ size_t dimZ() { return _data.dimZ(); }

			GpuDevice& gpu() const {
				throw "GPU property on CPU functor failed";
			}
		
		protected:
			AirFlowConsts _consts;
			DataKind _data;
				
			__device__ __host__ float& ux(size_t x, size_t y, size_t z) { return _data.ux(x, y, z); }
			__device__ __host__ float& uy(size_t x, size_t y, size_t z) { return _data.uy(x, y, z); }
			__device__ __host__ float& uz(size_t x, size_t y, size_t z) { return _data.uz(x, y, z); }
			__device__ __host__ float& ro(size_t x, size_t y, size_t z) { return _data.ro(x, y, z); }
			__device__ __host__ float& t(size_t x, size_t y, size_t z) { return _data.t(x, y, z); }
			__device__ __host__ float p(size_t x, size_t y, size_t z) { return _data.p(x, y, z); }
		
			__device__ __host__ float tau() { return _consts.Tau; }
			__device__ __host__ float h() { return _consts.H; }
			__device__ __host__ float nu() { return _consts.Nu; }
			__device__ __host__ float d() { return _consts.D; }
		};

		template<class DataKind>
		class Ro : public Base<DataKind> {
		public:
			__device__ __host__ Ro() { }
			Ro(AirFlowConsts consts, DataKind data)
				: Base<DataKind>(consts, data) { }

			__device__ __host__ bool IsZero(size_t x, size_t y, size_t z) {
				return (ux(x, y, z) == 0.0f && uy(x, y, z) == 0.0f && uz(x, y, z) == 0.0f);
			}

			__device__ __host__ float AlphaX(size_t x, size_t y, size_t z) {
				return ux(x, y, z) <= 0.0f
					? (- d() / pow(h(), 2.0f))
					: (- ux(x, y, z) / h()) - (d() / pow(h(), 2.0f));
			}

			__device__ __host__ float AlphaY(size_t x, size_t y, size_t z) {
				return uy(x, y, z) <= 0.0f
					? (- d() / pow(h(), 2.0f))
					: (- uy(x, y, z) / h()) - (d() / pow(h(), 2.0f));
			}

			__device__ __host__ float AlphaZ(size_t x, size_t y, size_t z) {
				return uz(x, y, z) <= 0.0f
					? (- d() / pow(h(), 2.0f))
					: (- uz(x, y, z) / h()) - (d() / pow(h(), 2.0f));
			}

			__device__ __host__ float BetaX(size_t x, size_t y, size_t z) {
				return ux(x, y, z) <= 0.0f
					? (- ux(x, y, z) / h()) + (1.0f / tau()) + (2.0f * d() / pow(h(), 2.0f))
					: (ux(x, y, z) / h()) + (1.0f / tau()) + (2.0f * d() / pow(h(), 2.0f));
			}

			__device__ __host__ float BetaY(size_t x, size_t y, size_t z) {
				return uy(x, y, z) <= 0.0f
					? (- uy(x, y, z) / h()) + (1.0f / tau()) + (2.0f * d() / pow(h(), 2.0f))
					: (uy(x, y, z) / h()) + (1.0f / tau()) + (2.0f * d() / pow(h(), 2.0f));
			}

			__device__ __host__ float BetaZ(size_t x, size_t y, size_t z) {
				return uz(x, y, z) <= 0.0f
					? (- uz(x, y, z) / h()) + (1.0f / tau()) + (2.0f * d() / pow(h(), 2.0f))
					: (uz(x, y, z) / h()) + (1.0f / tau()) + (2.0f * d() / pow(h(), 2.0f));
			}

			__device__ __host__ float GammaX(size_t x, size_t y, size_t z) {
				return ux(x, y, z) <= 0.0f
					? (ux(x, y, z) / h()) - (d() / pow(h(), 2.0f))
					: (- d() / pow(h(), 2.0f));
			}

			__device__ __host__ float GammaY(size_t x, size_t y, size_t z) {
				return uy(x, y, z) <= 0.0f
					? (uy(x, y, z) / h()) - (d() / pow(h(), 2.0f))
					: (- d() / pow(h(), 2.0f));
			}

			__device__ __host__ float GammaZ(size_t x, size_t y, size_t z) {
				return uz(x, y, z) <= 0.0f
					? (uz(x, y, z) / h()) - (d() / pow(h(), 2.0f))
					: (- d() / pow(h(), 2.0f));
			}

			__device__ __host__ float QX(size_t x, size_t y, size_t z) { return Q(x, y, z);	}
			__device__ __host__ float QY(size_t x, size_t y, size_t z) { return Q(x, y, z); }
			__device__ __host__ float QZ(size_t x, size_t y, size_t z) { return Q(x, y, z); }

			__device__ __host__ float getValue(size_t x, size_t y, size_t z) { return ro(x, y, z); }
			__device__ __host__ void setValue(size_t x, size_t y, size_t z, float value) { ro(x, y, z) = value; }

			__host__ __device__ float xMinBoundary(size_t y, size_t z, float boundaryValue, float preBoundaryValue) { return preBoundaryValue; }
			__host__ __device__ float xMaxBoundary(size_t y, size_t z, float boundaryValue, float preBoundaryValue) { return preBoundaryValue; }
			__host__ __device__ float yMinBoundary(size_t x, size_t z, float boundaryValue, float preBoundaryValue) { return preBoundaryValue; }
			__host__ __device__ float yMaxBoundary(size_t x, size_t z, float boundaryValue, float preBoundaryValue) { return preBoundaryValue; }
			__host__ __device__ float zMinBoundary(size_t x, size_t y, float boundaryValue, float preBoundaryValue) { return preBoundaryValue; }
			__host__ __device__ float zMaxBoundary(size_t x, size_t y, float boundaryValue, float preBoundaryValue) { return preBoundaryValue; }

		private:
			__device__ __host__ float Q(size_t x, size_t y, size_t z) { return (1.0f / tau()) * ro(x, y, z); }
		};

		template<class DataKind>
		class U : public Base<DataKind> {
		public:
			__device__ __host__ U() { }
			U(AirFlowConsts consts, DataKind data)
				: Base<DataKind>(consts, data) { }

			__device__ __host__ bool IsZero(size_t x, size_t y, size_t z) {
				return (ux(x, y, z) == 0.0f && uy(x, y, z) == 0.0f && uz(x, y, z) == 0.0f);
			}

			__device__ __host__ float AlphaX(size_t x, size_t y, size_t z) {
				return ux(x, y, z) <= 0.0f
					? - (nu() / pow(h(), 2.0f))
					:	- (ux(x, y, z) / h()) - (nu() / pow(h(), 2.0f));
			}

			__device__ __host__ float AlphaY(size_t x, size_t y, size_t z) {
				return uy(x, y, z) <= 0.0f
					? - (nu() / pow(h(), 2.0f))
					: - (uy(x, y, z) / h()) - (nu() / pow(h(), 2.0f));
			}

			__device__ __host__ float AlphaZ(size_t x, size_t y, size_t z) {
				return uz(x, y, z) <= 0.0f
					? - (nu() / pow(h(), 2.0f))
					: - (uz(x, y, z) / h()) - (nu() / pow(h(), 2.0f));
			}

			__device__ __host__ float BetaX(size_t x, size_t y, size_t z) {
				return ux(x, y, z) <= 0.0f
					? (- ux(x, y, z) / h()) + (1.0f / tau()) + (2.0f * nu() / pow(h(), 2.0f))
					: (ux(x, y, z) / h()) + (1.0f / tau()) + (2.0f * nu() / pow(h(), 2.0f));
			}

			__device__ __host__ float BetaY(size_t x, size_t y, size_t z) {
				return uy(x, y, z) <= 0.0f
					? (- uy(x, y, z) / h()) + (1.0f / tau()) + (2.0f * nu() / pow(h(), 2.0f))
					: (uy(x, y, z) / h()) + (1.0f / tau()) + (2.0f * nu() / pow(h(), 2.0f));
			}

			__device__ __host__ float BetaZ(size_t x, size_t y, size_t z) {
				return uz(x, y, z) <= 0.0f
					? (- uz(x, y, z) / h()) + (1.0f / tau()) + (2.0f * nu() / pow(h(), 2.0f))
					: (uz(x, y, z) / h()) + (1.0f / tau()) + (2.0f * nu() / pow(h(), 2.0f));
			}

			__device__ __host__ float GammaX(size_t x, size_t y, size_t z) {
				return ux(x, y, z) <= 0.0f
					? (ux(x, y, z) / h()) - (nu() / pow(h(), 2.0f))
					: - (nu() / pow(h(), 2.0f));
			}

			__device__ __host__ float GammaY(size_t x, size_t y, size_t z) {
				return uy(x, y, z) <= 0.0f
					? (uy(x, y, z) / h()) - (nu() / pow(h(), 2.0f))
					: - (nu() / pow(h(), 2.0f));
			}

			__device__ __host__ float GammaZ(size_t x, size_t y, size_t z) {
				return uz(x, y, z) <= 0.0f
					? (uz(x, y, z) / h()) - (nu() / pow(h(), 2.0f))
					: - (nu() / pow(h(), 2.0f));
			}

			__host__ __device__ float xMinBoundary(size_t y, size_t z, float boundaryValue, float preBoundaryValue) {
				//return (y > 5 && y <= 10 && z > 5 && z <= 10) ? preBoundaryValue : boundaryValue;
				return boundaryValue;
			}

			__host__ __device__ float xMaxBoundary(size_t y, size_t z, float boundaryValue, float preBoundaryValue) {
				//return (y > 5 && y <= 10 && z > 5 && z <= 10) ? preBoundaryValue : boundaryValue;
				return boundaryValue;
			}

			__host__ __device__ float yMinBoundary(size_t x, size_t z, float boundaryValue, float preBoundaryValue) { return boundaryValue; }
			__host__ __device__ float yMaxBoundary(size_t x, size_t z, float boundaryValue, float preBoundaryValue) { return boundaryValue; }
			__host__ __device__ float zMinBoundary(size_t x, size_t y, float boundaryValue, float preBoundaryValue) { return boundaryValue; }
			__host__ __device__ float zMaxBoundary(size_t x, size_t y, float boundaryValue, float preBoundaryValue) { return boundaryValue; }
		};

		template<class DataKind>
		class Ux : public U<DataKind> {
		public:
			__device__ __host__ Ux() { }
			Ux(AirFlowConsts consts, DataKind data)
				: U<DataKind>(consts, data) { }

			__device__ __host__ float QX(size_t x, size_t y, size_t z) {
				return (1.0f / tau()) * ux(x, y, z) -
					(1.0f / ro(x, y, z)) *
						((p(x+1,y+1,z+1) + p(x+1,y+1,z-1) + p(x+1,y-1,z+1) + p(x+1,y-1,z-1)
						- p(x-1,y+1,z+1) - p(x-1,y+1,z-1) - p(x-1,y-1,z+1) - p(x-1,y-1,z-1)) / (8.0f * h()));
			}

			__device__ __host__ float QY(size_t x, size_t y, size_t z) { return (1.0f / tau()) * ux(x, y, z); }
			__device__ __host__ float QZ(size_t x, size_t y, size_t z) { return (1.0f / tau()) * ux(x, y, z); }

			__device__ __host__ float getValue(size_t x, size_t y, size_t z) { return ux(x, y, z); }
			__device__ __host__ void setValue(size_t x, size_t y, size_t z, float value) { ux(x, y, z) = value; }
		};

		template<class DataKind>
		class Uy : public U<DataKind> {
		public:
			__device__ __host__ Uy() { }
			Uy(AirFlowConsts consts, DataKind data)
				: U<DataKind>(consts, data) { }

			__device__ __host__ float QX(size_t x, size_t y, size_t z) { return (1.0f / tau()) * uy(x, y, z); }

			__device__ __host__ float QY(size_t x, size_t y, size_t z) {
				return (1.0f / tau()) * uy(x, y, z) -
					(1.0f / ro(x, y, z)) *
						((p(x+1,y+1,z+1) + p(x+1,y+1,z-1) + p(x-1,y+1,z+1) + p(x-1,y+1,z-1)
						- p(x+1,y-1,z+1) - p(x+1,y-1,z-1) - p(x-1,y-1,z+1) - p(x-1,y-1,z-1)) / (8.0f * h()));
			}

			__device__ __host__ float QZ(size_t x, size_t y, size_t z) { return (1.0f / tau()) * uy(x, y, z); }

			__device__ __host__ float getValue(size_t x, size_t y, size_t z) { return uy(x, y, z); }
			__device__ __host__ void setValue(size_t x, size_t y, size_t z, float value) { uy(x, y, z) = value; }
		};

		template<class DataKind>
		class Uz : public U<DataKind> {
		public:
			__device__ __host__ Uz() { }
			Uz(AirFlowConsts consts, DataKind data)
				: U<DataKind>(consts, data) { }

			__device__ __host__ float QX(size_t x, size_t y, size_t z) { return (1.0f / tau()) * uz(x, y, z); }
			__device__ __host__ float QY(size_t x, size_t y, size_t z) { return (1.0f / tau()) * uz(x, y, z); }
			__device__ __host__ float QZ(size_t x, size_t y, size_t z) {
				return (1.0f / tau()) * uz(x, y, z) -
					(1.0f / ro(x, y, z)) *
						((p(x-1,y+1,z+1) + p(x-1,y-1,z+1) + p(x+1,y+1,z+1) + p(x+1,y-1,z+1)
						- p(x-1,y+1,z-1) - p(x-1,y-1,z-1) - p(x+1,y+1,z-1) - p(x+1,y-1,z-1)) / (8.0f * h()));
			}

			__device__ __host__ float getValue(size_t x, size_t y, size_t z) { return uz(x, y, z); }
			__device__ __host__ void setValue(size_t x, size_t y, size_t z, float value) { uz(x, y, z) = value; }
		};

		GpuDevice& Base<AirFlowDataD>::gpu() const {
			return _data.gpu();
		}
	}
}