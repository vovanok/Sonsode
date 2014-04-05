#pragma once

#include "HostData.hpp"
#include "Vectors.hpp"
#include "DeviceData.cu"

using Sonsode::HostData3D;
using Sonsode::DeviceData3D;
using Sonsode::GpuDevice;

struct FireSpreadConsts {
	float H; //��� �� ����������
	float Tau; //��� �� �������
	float Humidity; //���������
	float WindAngle; //���� ��������� �����
	float WindSpeed; //�������� �����
	float M2; //?
	float Danu; //?
	float TemOnBounds; //����������� �� ��������
	int IterFireBeginNum; //��������, �� ������� ���������� �������
	float Qbig; //?
	int Mstep; //?
	float Tzv; //?
	float TemKr; //?
	float Qlitl; //?
	float Ks; //?
	float EnviromentTemperature; //����������� ���������� �����

	float Hh;
	float Ap;
	Vector2D<float> WindU;
	float R0;

	FireSpreadConsts()
		: H(0.0f), Tau(0.0f), Humidity(0.0f), WindAngle(0.0f), WindSpeed(0.0f),
			M2(0.0f), Danu(0.0f), TemOnBounds(0.0f), IterFireBeginNum(0), Qbig(0.0f), Mstep(0), 
			Tzv(0.0f), TemKr(0.0f), Qlitl(0.0f), Ks(0.0f), EnviromentTemperature(0.0f) { }

	FireSpreadConsts(float h, float tau, float humidity, float windAngle, float windSpeed,
									 float m2, float danu, float temOnBounds, int iterFireBeginNum, float qbig,
									 int mstep, float tzv, float temKr, float qlitl, float ks, float enviromentTemperature)
		: H(h), Tau(tau), Humidity(humidity), WindAngle(windAngle), WindSpeed(windSpeed), M2(m2),
			Danu(danu), TemOnBounds(temOnBounds), IterFireBeginNum(iterFireBeginNum), Qbig(qbig), Mstep(mstep),
			Tzv(tzv), TemKr(temKr), Qlitl(qlitl), Ks(ks), EnviromentTemperature(enviromentTemperature) {

		Hh = pow(h, 2.0f);
		Ap = m2 / Hh;
		WindU.x = windSpeed * cos(windAngle);
		WindU.y = windSpeed * sin(windAngle);
		R0 = 1.0f / h;
	}
};

class FireSpreadDataH {
public:
	friend class FireSpreadDataD;

	size_t dimX() { return data.dimX(); }
	size_t dimY() { return data.dimY(); }

	FireSpreadDataH() { }
	FireSpreadDataH(size_t dimX, size_t dimY)
		: data(HostData3D<float>(dimX, dimY, 3)) { }

	void Erase() { data.Erase(); }
	void Fill(float value) { data.Fill(value); }
	
	float& t(size_t x, size_t y) { return data(x, y, 0); }
	float& roFuel(size_t x, size_t y) { return data(x, y, 1); }
	float& t4(size_t x, size_t y) { return data(x, y, 2); }

private:
	HostData3D<float> data;
};

class FireSpreadDataD {
public:
	__host__ __device__ size_t dimX() { return data.dimX(); }
	__host__ __device__ size_t dimY() { return data.dimY(); }

	__host__ __device__ FireSpreadDataD() { }

	FireSpreadDataD(const GpuDevice& gpu, FireSpreadDataH hostData)
		: data(DeviceData3D<float>(gpu, hostData.data)) { }

	void Erase() { data.Erase(); }
	void TakeFrom(const FireSpreadDataH& hostData) { data.TakeFrom(hostData.data); }
	void PutTo(FireSpreadDataH& hostData) { data.PutTo(hostData.data); }

	__device__ float& t(size_t x, size_t y) { return data(x, y, 0); }
	__device__ float& roFuel(size_t x, size_t y) { return data(x, y, 1); }
	__device__ float& t4(size_t x, size_t y) { return data(x, y, 2); }

	GpuDevice& gpu() const { return data.gpu(); }

private:
	DeviceData3D<float> data;
};