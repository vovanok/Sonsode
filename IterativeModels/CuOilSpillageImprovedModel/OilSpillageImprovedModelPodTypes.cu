#pragma once

#include "Vectors.hpp"
#include "HostData.hpp"
#include "DeviceData.cu"
#include "GpuDevice.hpp"

using Sonsode::HostData2D;
using Sonsode::DeviceData2D;
using Sonsode::HostData3D;
using Sonsode::DeviceData3D;
using Sonsode::GpuDevice;

struct OilSpillageConsts {
	Vector2D<float> Ukl;
	Vector2D<float> Beta;
	Vector2D<float> WindSpeed;
	float Temperature;
	float BackImpurity;
	float CoriolisFactor;
	float H;
	float Mult;
	float Sopr;
	float G;
	float Tok;
	float Tau;

	OilSpillageConsts()
		: Ukl(Vector2D<float>(0.0f, 0.0f)), Beta(Vector2D<float>(0.0f, 0.0f)),
			WindSpeed(Vector2D<float>(0.0f, 0.0f)), Temperature(0.0f), BackImpurity(0.0f),
			CoriolisFactor(0.0f), H(0.0f), Mult(0.0f), Sopr(0.0f), G(0.0f), Tok(0.0f), Tau(0.0f) { }

	OilSpillageConsts(Vector2D<float> ukl, Vector2D<float> beta,
			Vector2D<float> windSpeed, float temperature, float backImpurity,
			float coriolisFactor, float h, float mult, float sopr, float g, float tok, float tau)
		: Ukl(ukl), Beta(beta), WindSpeed(windSpeed), Temperature(temperature),
			BackImpurity(backImpurity), CoriolisFactor(coriolisFactor), H(h), Mult(mult),
			Sopr(sopr), G(g), Tok(tok), Tau(tau) { }
};

class OilSpillageDataH {
public:
	friend class OilSpillageDataD;

	size_t dimX() { return data.dimX(); }
	size_t dimY() { return data.dimY(); }

	OilSpillageDataH() { }
	OilSpillageDataH(size_t dimX, size_t dimY)
		: data(HostData3D<float>(dimX, dimY, 12)) { }

	void Erase() { data.Erase(); }
	void Fill(float value) { data.Fill(value); }

	void FillS(float value) {
		for (size_t z = 8; z <= 11; z++)
			for (size_t x = 0; x <= dimX(); x++)
				for (size_t y = 0; y <= dimY(); y++)
					data(x, y, z) = value;
	}

	float& w(size_t x, size_t y) { return data(x, y, 0); }
	float& waterUx(size_t x, size_t y) { return data(x, y, 1); }
	float& waterUy(size_t x, size_t y) { return data(x, y, 2); }
	float& oilUx(size_t x, size_t y) { return data(x, y, 3); }
	float& oilUy(size_t x, size_t y) { return data(x, y, 4); }
	float& deep(size_t x, size_t y) { return data(x, y, 5); }
	float& impurity(size_t x, size_t y) { return data(x, y, 6); }
	float& press(size_t x, size_t y) { return data(x, y, 7); }
	float& waterUxS(size_t x, size_t y) { return data(x, y, 8); }
	float& waterUyS(size_t x, size_t y) { return data(x, y, 9); }
	float& impurityS(size_t x, size_t y) { return data(x, y, 10); }
	float& pressS(size_t x, size_t y) { return data(x, y, 11); }

private:
	HostData3D<float> data;
};

class OilSpillageDataD {
public:
	__host__ __device__ size_t dimX() { return data.dimX(); }
	__host__ __device__ size_t dimY() { return data.dimY(); }

	__host__ __device__ OilSpillageDataD() { }

	OilSpillageDataD(const GpuDevice& gpu, size_t dimX, size_t dimY)
		: data(DeviceData3D<float>(gpu, dimX, dimY, 12)) { }

	OilSpillageDataD(const GpuDevice& gpu, OilSpillageDataH hostData)
		: data(DeviceData3D<float>(gpu, hostData.data)) { }

	void Erase() { data.Erase(); }
	void TakeFrom(const OilSpillageDataH& hostData) { data.TakeFrom(hostData.data); }
	void PutTo(OilSpillageDataH& hostData) { data.PutTo(hostData.data); }

	__device__ float& w(size_t x, size_t y) { return data(x, y, 0); }
	__device__ float& waterUx(size_t x, size_t y) { return data(x, y, 1); }
	__device__ float& waterUy(size_t x, size_t y) { return data(x, y, 2); }
	__device__ float& oilUx(size_t x, size_t y) { return data(x, y, 3); }
	__device__ float& oilUy(size_t x, size_t y) { return data(x, y, 4); }
	__device__ float& deep(size_t x, size_t y) { return data(x, y, 5); }
	__device__ float& impurity(size_t x, size_t y) { return data(x, y, 6); }
	__device__ float& press(size_t x, size_t y) { return data(x, y, 7); }
	__device__ float& waterUxS(size_t x, size_t y) { return data(x, y, 8); }
	__device__ float& waterUyS(size_t x, size_t y) { return data(x, y, 9); }
	__device__ float& impurityS(size_t x, size_t y) { return data(x, y, 10); }
	__device__ float& pressS(size_t x, size_t y) { return data(x, y, 11); }

	GpuDevice& gpu() const { return data.gpu(); }

private:
	DeviceData3D<float> data;
};