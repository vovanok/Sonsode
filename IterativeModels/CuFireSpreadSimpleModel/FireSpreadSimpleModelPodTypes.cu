#pragma once

#include "HostData.hpp"
#include "DeviceData.cu"

using Sonsode::HostData3D;
using Sonsode::DeviceData3D;
using Sonsode::GpuDevice;

struct FireSpreadConsts {
	float H; //Шаг по расстоянию
	float Tau; //Шаг по времени
	float Humidity; //Влажность
	float WindAngle; //Угол дуновения ветра
	float WindSpeed; //Скорость ветра
	float M2; //?
	float Danu; //?
	float TemOnBounds; //Температура на границах
	int IterFireBeginNum; //Итерация, на которой начинается горение
	float Qbig; //?
	int Mstep; //?
	float Tzv; //?
	float TemKr; //?
	float Qlitl; //?
	float Ks; //?
	float EnviromentTemperature; //Температура окружающей среды

	FireSpreadConsts()
		: H(0.0f), Tau(0.0f), Humidity(0.0f), WindAngle(0.0f), WindSpeed(0.0f),
			M2(0.0f), Danu(0.0f), TemOnBounds(0.0f), IterFireBeginNum(0), Qbig(0.0f), Mstep(0), 
			Tzv(0.0f), TemKr(0.0f), Qlitl(0.0f), Ks(0.0f), EnviromentTemperature(0.0f) { }

	FireSpreadConsts(float h, float tau, float humidity, float windAngle, float windSpeed,
									 float m2, float danu, float temOnBounds, int iterFireBeginNum, float qbig,
									 int mstep, float tzv, float temKr, float qlitl, float ks, float enviromentTemperature)
		: H(h), Tau(tau), Humidity(humidity), WindAngle(windAngle), WindSpeed(windSpeed), M2(m2),
			Danu(danu), TemOnBounds(temOnBounds), IterFireBeginNum(iterFireBeginNum), Qbig(qbig), Mstep(mstep),
			Tzv(tzv), TemKr(temKr), Qlitl(qlitl), Ks(ks), EnviromentTemperature(enviromentTemperature) { }
};

class FireSpreadDataH {
public:
	friend class FireSpreadDataD;

	size_t dimX() { return data.dimX(); }
	size_t dimY() { return data.dimY(); }

	FireSpreadDataH() { }
	FireSpreadDataH(size_t dimX, size_t dimY)
		: data(HostData3D<float>(dimX, dimY, 5)) { }

	void Erase() { data.Erase(); }
	void Fill(float value) { data.Fill(value); }
	
	float& t(size_t x, size_t y) { return data(x, y, 0); }
	float& roFuel(size_t x, size_t y) { return data(x, y, 1); }
	float& t4(size_t x, size_t y) { return data(x, y, 2); }
	float& q4(size_t x, size_t y) { return data(x, y, 3); }
	float& m(size_t x, size_t y) { return data(x, y, 4); }

private:
	HostData3D<float> data;
};

class FireSpreadDataD {
public:
	__host__ __device__ size_t dimX() { return data.dimX(); }
	__host__ __device__ size_t dimY() { return data.dimY(); }

	__host__ __device__ FireSpreadDataD() { }

	/*FireSpreadDataD(const GpuDevice& gpu, size_t dimX, size_t dimY)
		: data(DeviceData3D<float>(gpu, dimX, dimY, 5)) { }*/

	FireSpreadDataD(const GpuDevice& gpu, FireSpreadDataH hostData)
		: data(DeviceData3D<float>(gpu, hostData.data)) { }

	void Erase() { data.Erase(); }
	void TakeFrom(const FireSpreadDataH& hostData) { data.TakeFrom(hostData.data); }
	void PutTo(FireSpreadDataH& hostData) { data.PutTo(hostData.data); }

	__device__ float& t(size_t x, size_t y) { return data(x, y, 0); }
	__device__ float& roFuel(size_t x, size_t y) { return data(x, y, 1); }
	__device__ float& t4(size_t x, size_t y) { return data(x, y, 2); }
	__device__ float& q4(size_t x, size_t y) { return data(x, y, 3); }
	__device__ float& m(size_t x, size_t y) { return data(x, y, 4); }

	GpuDevice& gpu() const { return data.gpu(); }

private:
	DeviceData3D<float> data;
};