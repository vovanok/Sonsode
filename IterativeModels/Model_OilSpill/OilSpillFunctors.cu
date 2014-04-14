#pragma once

#include <math.h>
#include "Vectors.hpp"
#include "OilSpillPodTypes.cu"
#include "GpuDevice.hpp"
#include "GpuDeviceFactory.h"

namespace OilSpill {
	namespace Functors {
		using Sonsode::GpuDevice;
		using Sonsode::GpuDeviceFactory;
	
		template<class DataKind>
		class Base {
		public:
			__host__ __device__ Base() { }
			Base(OilSpillConsts consts, DataKind data)
				: _consts(consts), _data(data) { }

			__host__ __device__ size_t dimX() { return _data.dimX(); }
			__host__ __device__ size_t dimY() { return _data.dimY(); }

			GpuDevice& gpu() const {
				throw std::exception("GPU property on CPU functor failed");
			}

		protected:
			OilSpillConsts _consts;
			DataKind _data;

			__host__ __device__ float& w(size_t x, size_t y) { return _data.w(x, y); }
			__host__ __device__ float& waterUx(size_t x, size_t y) { return _data.waterUx(x, y); }
			__host__ __device__ float& waterUy(size_t x, size_t y) { return _data.waterUy(x, y); }
			__host__ __device__ float& oilUx(size_t x, size_t y) { return _data.oilUx(x, y); }
			__host__ __device__ float& oilUy(size_t x, size_t y) { return _data.oilUy(x, y); }
			__host__ __device__ float& deep(size_t x, size_t y) { return _data.deep(x, y); }
			__host__ __device__ float& impurity(size_t x, size_t y) { return _data.impurity(x, y); }
			__host__ __device__ float& press(size_t x, size_t y) { return _data.press(x, y); }

			__host__ __device__ float& waterUxS(size_t x, size_t y) { return _data.waterUxS(x, y); }
			__host__ __device__ float& waterUyS(size_t x, size_t y) { return _data.waterUyS(x, y); }
			__host__ __device__ float& impurityS(size_t x, size_t y) { return _data.impurityS(x, y); }
			__host__ __device__ float& pressS(size_t x, size_t y) { return _data.pressS(x, y); }

			__host__ __device__ Vector2D<float> ukl() { return _consts.Ukl; }
			__host__ __device__ Vector2D<float> beta() { return _consts.Beta; }
			__host__ __device__ Vector2D<float> windSpeed() { return _consts.WindSpeed; }
			__host__ __device__ float temperature() { return _consts.Temperature; }
			__host__ __device__ float backImpurity() { return _consts.BackImpurity; }
			__host__ __device__ float coriolisFactor() { return _consts.CoriolisFactor; }
			__host__ __device__ float h() { return _consts.H; }
			__host__ __device__ float mult() { return _consts.Mult; }
			__host__ __device__ float sopr() { return _consts.Sopr; }
			__host__ __device__ float g() { return _consts.G; }
			__host__ __device__ float tok() { return _consts.Tok; }
			__host__ __device__ float tau() { return _consts.Tau; }
		};

		template<class DataKind>
		class WaterU : public Base<DataKind> {
		public:
			__host__ __device__ WaterU() { }
			WaterU(OilSpillConsts consts, DataKind data)
				: Base<DataKind>(consts, data) { }

			__host__ __device__ bool IsZero(size_t x, size_t y) { return false; }

			__host__ __device__ float AlphaX(size_t x, size_t y) {
				return (-(waterUx(x, y) + abs(waterUx(x, y))) * (1.0f / (2.0f * h()))
					- (w(x, y) * (1.0f / pow(h(), 2.0f)) / (1.0f + 0.5f * h() * abs(waterUx(x, y)))));
			}

			__host__ __device__ float AlphaY(size_t x, size_t y) {
				return (-(waterUy(x, y) + abs(waterUy(x, y))) * (1.0f / (2.0f * h()))
					- (w(x, y) * (1.0f / pow(h(), 2.0f)) / (1.0f + 0.5f * h() * abs(waterUy(x, y)))));
			}

			__host__ __device__ float BetaX(size_t x, size_t y) {
				return (1.0f / tau()) + abs(waterUx(x, y)) * (1.0f / h())
					+ 2.0f * (w(x, y) * (1.0f / pow(h(), 2.0f)) / (1.0f + 0.5f * h() * abs(waterUx(x, y))));
			}

			__host__ __device__ float BetaY(size_t x, size_t y) {
				return (1.0f / tau()) + abs(waterUy(x, y)) * (1.0f / h())
					+ 2.0f * (w(x, y) * (1.0f / pow(h(), 2.0f)) / (1.0f + 0.5f * h() * abs(waterUy(x, y))));
			}

			__host__ __device__ float GammaX(size_t x, size_t y) {
				return (waterUx(x, y) - abs(waterUx(x, y))) * (1.0f / (2.0f * h()))
					- (w(x, y) * (1.0f / pow(h(), 2.0f)) / (1.0f + 0.5f * h() * abs(waterUx(x, y))));
			}

			__host__ __device__ float GammaY(size_t x, size_t y) {
				return (waterUy(x, y) - abs(waterUy(x, y))) * (1.0f / 2.0f * h())
					- (w(x, y) * (1.0f / pow(h(), 2.0f)) / (1.0f + 0.5f * h() * abs(waterUy(x, y))));
			}
		};

		template<class DataKind>
		class WaterUx : public WaterU<DataKind> {
		public:
			__host__ __device__ WaterUx() { }
			WaterUx(OilSpillConsts consts, DataKind data)
				: WaterU<DataKind>(consts, data) { }

			__host__ __device__ float getValue(size_t x, size_t y) { return waterUx(x, y); }
			__host__ __device__ void setValue(size_t x, size_t y, float value) { waterUx(x, y) = value; }
		
			__host__ __device__ float QX(size_t x, size_t y);
			__host__ __device__ float QY(size_t x, size_t y);
		};

		template<class DataKind>
		class WaterUy : public WaterU<DataKind> {
		public:
			__host__ __device__ WaterUy() { }
			WaterUy(OilSpillConsts consts, DataKind data)
				: WaterU<DataKind>(consts, data) { }

			__host__ __device__ float getValue(size_t x, size_t y) { return waterUy(x, y); }
			__host__ __device__ void setValue(size_t x, size_t y, float value) { waterUy(x, y) = value; }

			__host__ __device__ float QX(size_t x, size_t y);
			__host__ __device__ float QY(size_t x, size_t y);
		};

		template<class DataKind>
		class ImpurityAndPress : public Base<DataKind> {
		public:
			__host__ __device__ ImpurityAndPress() { }
			ImpurityAndPress(OilSpillConsts consts, DataKind data)
				: Base<DataKind>(consts, data) { }

			__host__ __device__ bool IsZero(size_t x, size_t y) { return false; }
			__host__ __device__ float AlphaX(size_t x, size_t y) { return Alpha(oilUx(x, y), x, y); }
			__host__ __device__ float AlphaY(size_t x, size_t y) { return Alpha(oilUy(x, y), x, y); }
			__host__ __device__ float BetaX(size_t x, size_t y) { return Beta(oilUx(x, y), x, y); }
 			__host__ __device__ float BetaY(size_t x, size_t y) { return Beta(oilUy(x, y), x, y); }
			__host__ __device__ float GammaX(size_t x, size_t y) { return Gamma(oilUx(x, y), x, y); }
			__host__ __device__ float GammaY(size_t x, size_t y) { return Gamma(oilUy(x, y), x, y); }

			__host__ __device__ float xMinBoundary(size_t y, float boundaryValue, float preBoundaryValue) { return preBoundaryValue; }
			__host__ __device__ float xMaxBoundary(size_t y, float boundaryValue, float preBoundaryValue) { return preBoundaryValue; }
			__host__ __device__ float yMinBoundary(size_t x, float boundaryValue, float preBoundaryValue) { return preBoundaryValue; }
			__host__ __device__ float yMaxBoundary(size_t x, float boundaryValue, float preBoundaryValue) { return preBoundaryValue; }

		private:
			__host__ __device__ float Alpha(float uSelf, size_t x, size_t y) {
				return (-(uSelf + abs(uSelf)) / (2.0f * h())
					- w(x, y) / pow(h(), 2.0f) / (1.0f + 0.5f * h() * abs(uSelf)));
			}

			__host__ __device__ float Beta(float uSelf, size_t x, size_t y) {
				return 1.0f / tau() + abs(uSelf) / h()
					+ 2.0f * (w(x, y) / pow(h(), 2.0f) / (1.0f + 0.5f * h() * abs(uSelf)));
			}

			__host__ __device__ float Gamma(float uSelf, size_t x, size_t y) {
				return (uSelf - abs(uSelf)) / (2.0f * h())
					- w(x, y) / pow(h(), 2.0f) / (1.0f + 0.5f * h() * abs(uSelf));
			}
		};

		template<class DataKind>
		class Impurity : public ImpurityAndPress<DataKind> {
		public:
			__host__ __device__ Impurity() { }
			Impurity(OilSpillConsts consts, DataKind data)
				: ImpurityAndPress<DataKind>(consts, data) { }

			__host__ __device__ float getValue(size_t x, size_t y) { return impurity(x, y); }
			__host__ __device__ void setValue(size_t x, size_t y, float value) { impurity(x, y) = value; }

			__host__ __device__ float QX(size_t x, size_t y);
			__host__ __device__ float QY(size_t x, size_t y);
		};

		template<class DataKind>
		class Press : public ImpurityAndPress<DataKind> {
		public:
			__host__ __device__ Press() { }
			Press(OilSpillConsts consts, DataKind data)
				: ImpurityAndPress<DataKind>(consts, data) { }

			__host__ __device__ float getValue(size_t x, size_t y) { return press(x, y); }
			__host__ __device__ void setValue(size_t x, size_t y, float value) { press(x, y) = value; }
		
			__host__ __device__ float QX(size_t x, size_t y);
			__host__ __device__ float QY(size_t x, size_t y);
		};

		template<class DataKind>
		class WaterUxS : public Base<DataKind> {
		public:
			__host__ __device__ WaterUxS() { }
			WaterUxS(OilSpillConsts consts, DataKind data)
				: Base<DataKind>(consts, data) { }

			__host__ __device__ float getValue(size_t x, size_t y) { return waterUxS(x, y); }
			__host__ __device__ void setValue(size_t x, size_t y, float value) { waterUxS(x, y) = value; }

			__host__ __device__ float Formula(size_t x, size_t y, float s, float l , float r, float u, float d) {
				return -(((press(x, y) + press(x, y-1)) - (press(x-1, y) + press(x-1, y-1))) / h());
			}
		};

		template<class DataKind>
		class WaterUyS : public Base<DataKind> {
		public:
			__host__ __device__ WaterUyS() { }
			WaterUyS(OilSpillConsts consts, DataKind data)
				: Base<DataKind>(consts, data) { }

			__host__ __device__ float getValue(size_t x, size_t y) { return waterUyS(x, y); }
			__host__ __device__ void setValue(size_t x, size_t y, float value) { waterUyS(x, y) = value; }

			__host__ __device__ float Formula(size_t x, size_t y, float s, float l , float r, float u, float d) {
				return -(((press(x-1, y) + press(x, y)) - (press(x-1, y-1) + press(x, y-1))) / h());
			}
		};

		template<class DataKind>
		class WaterUx_Complete : public Base<DataKind> {
		public:
			__host__ __device__ WaterUx_Complete() { }
			WaterUx_Complete(OilSpillConsts consts, DataKind data)
				: Base<DataKind>(consts, data) { }

			__host__ __device__ float getValue(size_t x, size_t y) { return waterUx(x, y); }
			__host__ __device__ void setValue(size_t x, size_t y, float value) { waterUx(x, y) = value; }

			__host__ __device__ float Formula(size_t x, size_t y, float s, float l , float r, float u, float d) {
				return (((s + ukl().x * tau())
						/ (1.0f + sopr() / (1.0f + deep(x, y)) * tau()) + beta().x * oilUx(x, y))
						/ (1.0f + beta().x * tau()))
						- coriolisFactor() * waterUy(x, y) * tau();
			}
		};

		template<class DataKind>
		class WaterUy_Complete : public Base<DataKind> {
		public:
			__host__ __device__ WaterUy_Complete() { }
			WaterUy_Complete(OilSpillConsts consts, DataKind data)
				: Base<DataKind>(consts, data) { }

			__host__ __device__ float getValue(size_t x, size_t y) { return waterUy(x, y); }
			__host__ __device__ void setValue(size_t x, size_t y, float value) { waterUy(x, y) = value; }

			__host__ __device__ float Formula(size_t x, size_t y, float s, float l , float r, float u, float d) {
				return (((s + ukl().y * tau())
						/ (1.0f + sopr() / (1.0f + deep(x, y)) * tau()) + beta().x * oilUy(x, y))
						/ (1.0f + beta().x * tau()))
						+ coriolisFactor() * waterUx(x, y) * tau();
			}
		};

		template<class DataKind>
		class OilUx : public Base<DataKind> {
		public:
			__host__ __device__ OilUx() { }
			OilUx(OilSpillConsts consts, DataKind data)
				: Base<DataKind>(consts, data) { }

			__host__ __device__ float getValue(size_t x, size_t y) { return oilUx(x, y); }
			__host__ __device__ void setValue(size_t x, size_t y, float value) { oilUx(x, y) = value; }

			__host__ __device__ float Formula(size_t x, size_t y, float s, float l , float r, float u, float d) {
				return (-g() * tok() * (impurity(x+1, y) - impurity(x-1, y)) / 2.0f / h()
					+ beta().x * waterUx(x, y) + beta().y * windSpeed().x) / (beta().x + beta().y);
				//md->get(x, y).OilUx = 1.1f * md->get(x, y).WaterUx + 0.03f * md->WindSpeedX;
			}
		};

		template<class DataKind>
		class OilUy : public Base<DataKind> {
		public:
			__host__ __device__ OilUy() { }
			OilUy(OilSpillConsts consts, DataKind data)
				: Base<DataKind>(consts, data) { }

			__host__ __device__ float getValue(size_t x, size_t y) { return oilUy(x, y); }
			__host__ __device__ void setValue(size_t x, size_t y, float value) { oilUy(x, y) = value; }

			__host__ __device__ float Formula(size_t x, size_t y, float s, float l , float r, float u, float d) {
				return (-g() * tok() * (impurity(x, y+1) - impurity(x, y-1)) / 2.0f / h()
					+ beta().x * waterUy(x, y) + beta().y * windSpeed().y) / (beta().x + beta().y);
				//md->get(x, y).OilUy = 1.1f * md->get(x, y).WaterUy + 0.03f * md->WindSpeedY;
			}
		};

		template<class DataKind>
		class PressS : public Base<DataKind> {
		public:
			__host__ __device__ PressS() { }
			PressS(OilSpillConsts consts, DataKind data)
				: Base<DataKind>(consts, data) { }

			__host__ __device__ float getValue(size_t x, size_t y) { return pressS(x, y); }
			__host__ __device__ void setValue(size_t x, size_t y, float value) { pressS(x, y) = value; }
		
			__host__ __device__ float Formula(size_t x, size_t y, float s, float l , float r, float u, float d) {
				return -1.5f * (mult() + s)
						* ((waterUx(x+1, y+1) + waterUx(x+1, y))
						- (waterUx(x, y) + waterUx(x, y+1))
						+ (waterUy(x, y+1) + waterUy(x+1, y+1))
						- (waterUy(x, y) + waterUy(x+1, y))) / (2.0f * h());
			}

			__host__ __device__ float xMinBoundary(size_t y, float boundaryValue, float preBoundaryValue) { return boundaryValue; }
			__host__ __device__ float xMaxBoundary(size_t y, float boundaryValue, float preBoundaryValue) {	return boundaryValue; }
			__host__ __device__ float yMinBoundary(size_t x, float boundaryValue, float preBoundaryValue) { return boundaryValue; }
			__host__ __device__ float yMaxBoundary(size_t x, float boundaryValue, float preBoundaryValue) { return preBoundaryValue; }
		};

		template<class DataKind>
		class ImpurityS : public Base<DataKind> {
		public:
			__host__ __device__ ImpurityS() { }
			ImpurityS(OilSpillConsts consts, DataKind data)
				: Base<DataKind>(consts, data) { }

			__host__ __device__ float getValue(size_t x, size_t y) { return impurityS(x, y); }
			__host__ __device__ void setValue(size_t x, size_t y, float value) { impurityS(x, y) = value; }

			__host__ __device__ float xMinBoundary(size_t y, float boundaryValue, float preBoundaryValue) { return boundaryValue; }
			__host__ __device__ float xMaxBoundary(size_t y, float boundaryValue, float preBoundaryValue) { return preBoundaryValue; }
			__host__ __device__ float yMinBoundary(size_t x, float boundaryValue, float preBoundaryValue) { return boundaryValue; }
			__host__ __device__ float yMaxBoundary(size_t x, float boundaryValue, float preBoundaryValue) { return boundaryValue; }
		};

		template<class DataKind>
		class W : public Base<DataKind> {
		public:
			__host__ __device__ W() { }
			W(OilSpillConsts consts, DataKind data)
				: Base<DataKind>(consts, data) { }

			__host__ __device__ float getValue(size_t x, size_t y) { return w(x, y); }
			__host__ __device__ void setValue(size_t x, size_t y, float value) { w(x, y) = value; }
		
			__host__ __device__ float Formula(size_t x, size_t y, float s, float l , float r, float u, float d) {
				return 0.16f * deep(x, y) * sqrt(pow(waterUx(x, y), 2.0f) + pow(waterUy(x, y), 2.0f));
			}
		};

		template<class DataKind>
		class ImpurityIstok : public Base<DataKind> {
		public:
			__host__ __device__ ImpurityIstok() { }
			ImpurityIstok(OilSpillConsts consts, DataKind data)
				: Base<DataKind>(consts, data) { }

			__host__ __device__ float getValue(size_t x, size_t y) { return impurity(x, y); }
			__host__ __device__ void setValue(size_t x, size_t y, float value) { impurity(x, y) = value; }

			__host__ __device__ float Formula(size_t x, size_t y, float s, float l , float r, float u, float d) {
				return backImpurity() + (impurity(x, y) - backImpurity())
						/ (1.0f + (1.15f * exp(0.05f * (temperature() - 20.0f)) * (1.0f / 24.0f / 3600.0f)) * tau());
			}
		};

		template<class DataKind>
		class IslandResolver : public Base<DataKind> {
		public:
			__host__ __device__ IslandResolver() { }
			IslandResolver(OilSpillConsts consts, DataKind data)
				: Base<DataKind>(consts, data) { }

			__host__ __device__ void Action(size_t x, size_t y) {
				if (deep(x, y) <= 0.0f) {
					waterUx(x, y) = 0.0f;
					waterUy(x, y) = 0.0f;

					oilUx(x, y) = 0.0f;
					oilUy(x, y) = 0.0f;

					impurity(x, y) = 0.0f;
				}
			}
		};

		#pragma region Base

		GpuDevice& Base<OilSpillDataD>::gpu() const {
			return _data.gpu();
		}

		#pragma endregion

		#pragma region Ux_functor

		template<class DataKind>
		float WaterUx<DataKind>::QX(size_t x, size_t y) {
			float dobY = (w(x, y) * (1.0f / pow(h(), 2.0f))
				/ (1.0f + 0.5f * h() * abs(waterUy(x, y))));

			float dX = waterUx(x, y) <= 0.0f
				? (w(x + 1, y) - w(x, y)) / h()
				: (w(x, y) - w(x - 1, y)) / h();

			float uX = waterUx(x, y) <= 0.0f
				? (waterUx(x + 1, y) - waterUx(x, y)) / h()
				: (waterUx(x, y) - waterUx(x - 1, y)) / h();

			return ((waterUx(x, y) + abs(waterUy(x, y))) / (2.0f * h()) + dobY)
				* waterUx(x, y - 1)
				+ (1.0f / tau() - abs(waterUy(x, y)) * (1.0f / h()) - 2.0f * dobY)
				* waterUx(x, y - 1)
				+ ((-waterUx(x, y) + abs(waterUy(x, y))) * (1.0f / (2.0f * h())) + dobY)
				* waterUx(x, y + 1) + dX * uX * dobY + waterUxS(x, y) / 2.0f;
		}

		template<class DataKind>
		float WaterUx<DataKind>::QY(size_t x, size_t y) {
			float dobX = w(x, y) * (1.0f / pow(h(), 2.0f))
				/ (1.0f + 0.5f * h() * abs(waterUx(x, y)));

			float dY = waterUy(x, y) <= 0.0f
				? (w(x, y + 1) - w(x, y)) / h()
				: (w(x, y) - w(x, y - 1)) / h();

			float uY = waterUy(x, y) <= 0.0f
				? (waterUx(x, y + 1) - waterUx(x, y)) / h()
				: (waterUx(x, y) - waterUx(x, y - 1)) / h();

			return ((waterUy(x, y) + abs(waterUx(x, y))) / (2.0f * h()) + dobX)
				* waterUx(x - 1, y)
				+ (1.0f / tau() - abs(waterUx(x, y)) / h() - 2.0f * dobX)
				* waterUx(x, y)
				+ ((-waterUy(x, y) + abs(waterUx(x, y))) / (2.0f * h()) + dobX)
				* waterUx(x + 1, y)	+ dY * uY * dobX + waterUxS(x, y) / 2.0f;
		}

		#pragma endregion

		#pragma region Uy_functor

		template<class DataKind>
		float WaterUy<DataKind>::QX(size_t x, size_t y) {
			float dobY = (w(x, y) * (1.0f / pow(h(), 2.0f))
				/ (1.0f + 0.5f * h() * abs(waterUy(x, y))));

			float dX = waterUx(x, y) <= 0.0f
				? (w(x+1, y) - w(x, y)) / h()
				: (w(x, y) - w(x-1, y)) / h();

			float uX = waterUx(x, y) <= 0.0f
				? (waterUy(x+1, y) - waterUy(x, y)) / h()
				: (waterUy(x, y) - waterUy(x-1, y)) / h();

			return ((waterUx(x, y) + abs(waterUy(x, y))) * (1.0f / (2.0f * h())) + dobY)
				* waterUy(x, y-1)
				+ (1.0f / tau() - abs(waterUy(x, y)) * (1.0f / h()) - 2.0f * dobY)
				* waterUy(x, y-1)
				+ ((-waterUx(x, y) + abs(waterUy(x, y))) * (1.0f / (2.0f * h())) + dobY)
				* waterUy(x, y+1) + dX * uX *dobY + waterUyS(x, y) / 2.0f;
		}

		template<class DataKind>
		float WaterUy<DataKind>::QY(size_t x, size_t y) {
			float dobX = w(x, y) * (1.0f / pow(h(), 2.0f))
				/ (1.0f + 0.5f * h() * abs(waterUx(x, y)));

			float dY = waterUy(x, y) <= 0.0f
				? (w(x, y+1) - w(x, y)) * (1.0f / h())
				: (w(x, y) - w(x, y-1)) * (1.0f / h());

			float uY = waterUy(x, y) <= 0.0f
				? (waterUy(x, y+1) - waterUy(x, y)) / h()
				: (waterUy(x, y) - waterUy(x, y-1)) / h();

			return ((waterUy(x, y) + abs(waterUx(x, y))) / h() + dobX)
				* waterUy(x-1, y)
				+ (1.0f / tau() - abs(waterUx(x, y)) / h() - 2.0f * dobX)
				* waterUy(x, y)
				+ ((-waterUy(x, y) + abs(waterUx(x, y))) / (2.0f * h()) + dobX)
				* waterUy(x+1, y)
				+ dY * uY * dobX + waterUyS(x, y) / 2.0f;
		}

		#pragma endregion

		#pragma region Impurity_functor

		template<class DataKind>
		float Impurity<DataKind>::QX(size_t x, size_t y) {
			float wX = oilUx(x, y) <= 0.0f
				? (w(x + 1, y) - w(x, y)) / h()
				: (w(x, y) - w(x - 1, y)) / h();

			float impurityX = oilUx(x, y) <= 0.0f
				? (impurity(x + 1, y) - impurity(x, y)) / h()
				: (impurity(x, y) - impurity(x - 1, y)) / h();

			float dobY = w(x, y) / pow(h(), 2.0f)
				/ (1.0f + 0.5f * h() * abs(oilUy(x, y)));

			return ((oilUx(x, y) + abs(oilUy(x, y))) / (2.0f * h()) + dobY)
				* impurity(x, y - 1)
				+ (1.0f / tau() - abs(oilUy(x, y)) / h() - 2.0f * dobY)
				* impurity(x, y)
				+ ((-oilUx(x, y) + abs(oilUy(x, y))) / (2.0f * h()) + dobY)
				* impurity(x, y + 1)
				+ wX * impurityX * dobY + impurityS(x, y) / 2.0f;
		}

		template<class DataKind>
		float Impurity<DataKind>::QY(size_t x, size_t y) {
			float wY = oilUy(x, y) <= 0.0f
				? (w(x, y+1) - w(x, y)) / h()
				: (w(x, y) - w(x, y-1)) / h();

			float impurityY = oilUy(x, y) <= 0.0f
				? (impurity(x, y + 1) - impurity(x, y)) / h()
				: (impurity(x, y) - impurity(x, y - 1)) / h();

			float dobX = w(x, y) / pow(h(), 2.0f)
				/ (1.0f + 0.5f * h() * abs(oilUx(x, y)));

			return ((oilUy(x, y) + abs(oilUx(x, y))) / (2.0f * h()) + dobX)
				* impurity(x - 1, y)
				+ (1.0f / tau() - abs(oilUx(x, y)) / h() - 2.0f * dobX)
				* impurity(x, y)
				+ ((-oilUy(x, y) + abs(oilUx(x, y))) / (2.0f * h()) + dobX)
				* impurity(x+1, y)
				+ wY * impurityY * dobX + impurityS(x, y) / 2.0f;
		}

		#pragma endregion

		#pragma region Press_functor

		template<class DataKind>
		float Press<DataKind>::QX(size_t x, size_t y) {
			float wX = oilUx(x, y) <= 0.0f
				? (w(x+1, y) - w(x, y)) / h()
				: (w(x, y) - w(x-1, y)) / h();

			float pressX = oilUx(x, y) <= 0.0f
				? (press(x+1, y) - press(x, y)) / h()
				: (press(x, y) - press(x-1, y)) / h();

			float dobY = w(x, y) / pow(h(), 2.0f)
				/ (1.0f + 0.5f * h() * abs(oilUy(x, y)));

			return ((oilUx(x, y) + abs(oilUy(x, y))) / (2.0f * h()) + dobY)
				* press(x, y-1)
				+ (1.0f / tau() - abs(oilUy(x, y)) / h() - 2.0f * dobY)
				* press(x, y)
				+ ((-oilUx(x, y) + abs(oilUy(x, y))) / (2.0f * h()) + dobY)
				* press(x, y+1)
				+ wX * pressX * dobY + pressS(x, y) / 2.0f;
		}

		template<class DataKind>
		float Press<DataKind>::QY(size_t x, size_t y) {
			float wY = oilUy(x, y) <= 0.0f
				? (w(x, y+1) - w(x, y)) / h()
				: (w(x, y) - w(x, y-1)) / h();

			float pressY = oilUy(x, y) <= 0.0f
				? (press(x, y+1) - press(x, y)) / h()
				: (press(x, y) - press(x, y - 1)) / h();

			float dobX = w(x, y) / pow(h(), 2.0f)
				/ (1.0f + 0.5f * h() * abs(oilUx(x, y)));

			return ((oilUy(x, y) + abs(oilUx(x, y))) / (2.0f * h()) + dobX)
				* press(x-1, y)
				+ (1.0f / tau() - abs(oilUx(x, y)) / h() - 2.0f * dobX)
				* press(x, y)
				+ ((-oilUy(x, y) + abs(oilUx(x, y))) / (2.0f * h()) + dobX)
				* press(x+1, y)
				+ wY * pressY * dobX + pressS(x, y) / 2.0f;
		}

		#pragma endregion
	}
}