#pragma once

#include "HostData.hpp"
#include "DeviceData.cu"

using Sonsode::HostData3D;
using Sonsode::DeviceData3D;
using Sonsode::HostData4D;
using Sonsode::DeviceData4D;
using Sonsode::GpuDevice;

#define R_GAS_CONST 8.3144621f //Дж / (моль * К)
const float Tenviroment = 273.0f + 20.0f; //К

__device__ __host__ static inline float GetP(float ro, float t) {
	return R_GAS_CONST * ro * t;
		//Sigma * pow(Data->at(x, y, z).Ro, S);
}

//struct AirFlowPoint {
//	float Ux, Uy, Uz, Ro, T, P;
//	
//	AirFlowPoint() : Ux(0.0f), Uy(0.0f), Uz(0.0f), Ro(0.0f), T(0.0f), P(GetP(0.0f, 0.0f)) { }
//	AirFlowPoint(float ux, float uy, float uz, float ro, float t) : Ux(ux), Uy(uy), Uz(uz), Ro(ro), T(t), P(GetP(ro, t)) { }
//	
//	/*std::string ToString() const {
//		std::stringstream ss;
//		ss << std::fixed << std::setprecision(2) << Ro << ";" << Ux << ";" << Uy << ";" << Uz;
//		return ss.str();
//	}*/
//};

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

//enum class BorderKind {
//	BORDERKIND_NONE,
//	BORDERKIND_FIRST,
//	BORDERKIND_SECOND
//};

//struct SpecialAirArea {
//	size_t X0, Y0, Z0;
//	size_t DimX, DimY, DimZ;
//	BorderKind xMinBorder;
//	BorderKind xMaxBorder;
//	BorderKind yMinBorder;
//	BorderKind yMaxBorder;
//	BorderKind zMinBorder;
//	BorderKind zMaxBorder;
//
//	AirFlowPoint SpecialValue;
//
//	SpecialAirArea(size_t x0, size_t y0, size_t z0, size_t dimX, size_t dimY, size_t dimZ, AirFlowPoint specialValue)
//		: X0(x0), Y0(y0), Z0(z0), DimX(dimX), DimY(dimY), DimZ(dimZ), SpecialValue(specialValue),
//			xMinBorder(BorderKind::BORDERKIND_NONE), xMaxBorder(BorderKind::BORDERKIND_NONE),
//			yMinBorder(BorderKind::BORDERKIND_NONE), yMaxBorder(BorderKind::BORDERKIND_NONE),
//			zMinBorder(BorderKind::BORDERKIND_NONE), zMaxBorder(BorderKind::BORDERKIND_NONE) { }
//
//	bool IsInside(size_t x, size_t y, size_t z) {
//		return (X0 <= x) && (x < X0 + DimX)
//				&& (Y0 <= y) && (y < Y0 + DimY)
//				&& (Z0 <= z) && (z < Z0 + DimZ);
//	}
//
//	BorderKind XminBorder(size_t x, size_t y, size_t z) {
//		if ((x == X0) && IsInside(x, y, z))
//			return xMinBorder;
//		return BorderKind::BORDERKIND_NONE;
//	}
//
//	BorderKind XmaxBorder(size_t x, size_t y, size_t z) {
//		if ((x == X0 + DimX - 1) && IsInside(x, y, z))
//			return xMaxBorder;
//		return BorderKind::BORDERKIND_NONE;
//	}
//
//	BorderKind YminBorder(size_t x, size_t y, size_t z) {
//		if ((y == Y0) && IsInside(x, y, z))
//			return yMinBorder;
//		return BorderKind::BORDERKIND_NONE;
//	}
//
//	BorderKind YmaxBorder(size_t x, size_t y, size_t z) {
//		if ((y == Y0 + DimY - 1) && IsInside(x, y, z))
//			return yMaxBorder;
//		return BorderKind::BORDERKIND_NONE;
//	}
//
//	BorderKind ZminBorder(size_t x, size_t y, size_t z) {
//		if ((z == Z0) && IsInside(x, y, z))
//			return zMinBorder;
//		return BorderKind::BORDERKIND_NONE;
//	}
//
//	BorderKind ZmaxBorder(size_t x, size_t y, size_t z) {
//		if ((z == Z0 + DimZ - 1) && IsInside(x, y, z))
//			return zMaxBorder;
//		return BorderKind::BORDERKIND_NONE;
//	}
//};