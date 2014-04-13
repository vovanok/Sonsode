#pragma once

#include "HostData.hpp"
#include "DeviceData.cu"

namespace AirFlow {
	using Sonsode::HostData3D;
	using Sonsode::DeviceData3D;
	using Sonsode::HostData4D;
	using Sonsode::DeviceData4D;
	using Sonsode::GpuDevice;

	#define R_GAS_CONST 8.3144621f //Дж / (моль * К)

	__device__ __host__ static inline float GetP(float ro, float t) {
		return R_GAS_CONST * ro * t;
	}

	struct AirFlowConsts {
		float Tau;
		float H;
		float Nu;
		float D;

		AirFlowConsts()
			: Tau(0.0f), H(0.0f), Nu(0.0f), D(0.0f) { }

		AirFlowConsts(float tau, float h, float nu, float d)
			: Tau(tau), H(h), Nu(nu), D(d) { }
	};

	class AirFlowDataH {
	public:
		friend class AirFlowDataD;

		AirFlowDataH() { }
		AirFlowDataH(size_t dimX, size_t dimY, size_t dimZ)
			: data(HostData4D<float>(dimX, dimY, dimZ, 5)) { }

		size_t dimX() { return data.dimX(); }
		size_t dimY() { return data.dimY(); }
		size_t dimZ() { return data.dimZ(); }

		void Erase() { data.Erase(); }
		void Fill(float value) { data.Fill(value); }

		float& ux(size_t x, size_t y, size_t z) { return data(x, y, z, 0); }
		float& uy(size_t x, size_t y, size_t z) { return data(x, y, z, 1); }
		float& uz(size_t x, size_t y, size_t z) { return data(x, y, z, 2); }
		float& ro(size_t x, size_t y, size_t z) { return data(x, y, z, 3); }
		float& t(size_t x, size_t y, size_t z) { return data(x, y, z, 4); }
		float p(size_t x, size_t y, size_t z) { return GetP(ro(x, y, z), t(x, y, z)); }

	private:
		HostData4D<float> data;
	};

	class AirFlowDataD {
	public:
		__device__ __host__ AirFlowDataD() { }
		AirFlowDataD(const GpuDevice& gpu, size_t dimX, size_t dimY, size_t dimZ)
			: data(DeviceData4D<float>(gpu, dimX, dimY, dimZ, 5)) { }
		AirFlowDataD(const GpuDevice& gpu, const AirFlowDataH& hostData)
			: data(DeviceData4D<float>(gpu, hostData.data)) { }

		__device__ __host__ size_t dimX() { return data.dimX(); }
		__device__ __host__ size_t dimY() { return data.dimY(); }
		__device__ __host__ size_t dimZ() { return data.dimZ(); }

		void Erase() { data.Erase(); }
		void TakeFrom(const AirFlowDataH& hostData) { data.TakeFrom(hostData.data); }
		void PutTo(AirFlowDataH& hostData) { data.PutTo(hostData.data); }

		__device__ float& ux(size_t x, size_t y, size_t z) { return data(x, y, z, 0); }
		__device__ float& uy(size_t x, size_t y, size_t z) { return data(x, y, z, 1); }
		__device__ float& uz(size_t x, size_t y, size_t z) { return data(x, y, z, 2); }
		__device__ float& ro(size_t x, size_t y, size_t z) { return data(x, y, z, 3); }
		__device__ float& t(size_t x, size_t y, size_t z) { return data(x, y, z, 4); }
		__device__ float p(size_t x, size_t y, size_t z) { return GetP(ro(x, y, z), t(x, y, z)); }

		GpuDevice& gpu() const { return data.gpu(); }

	private:
		DeviceData4D<float> data;
	};
}