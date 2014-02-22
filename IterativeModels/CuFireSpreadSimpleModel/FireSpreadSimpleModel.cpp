#include "FireSpreadSimpleModel.h"

FireSpreadSimpleModel::FireSpreadSimpleModel(FireSpreadConsts consts, FireSpreadDataH data)
		: IterativeModel(consts.Tau), _consts(consts), _data(data) {

	AddCalculationMethod("cpu", std::bind(std::mem_fun(&FireSpreadSimpleModel::CalculationMethod_CPU), this));
	AddCalculationMethod("gpu", std::bind(std::mem_fun(&FireSpreadSimpleModel::CalculationMethod_GPU), this));

	hh = pow(consts.H, 2.0f);
	ap = consts.M2 / hh;
	u.x = consts.WindSpeed * cos(consts.WindAngle);
	u.y = consts.WindSpeed * sin(consts.WindAngle);
	r0 = 1.0f / consts.H;

	lx = HostData1D<float>(data.dimY());
	lx.Fill(0.0f);
	ly = HostData1D<float>(data.dimX());
	ly.Fill(0.0f);

	for (size_t y = 0; y < data.dimY(); y++) {
		if (2 < y && y <= data.dimY()-2)
			lx(y) = ap / (1.0f + 2.0f * ap - ap * lx(y-1));

		for (size_t x = 0; x < data.dimX(); x++) {
			if (2 < y && y <= data.dimY()-2)
				ly(x) = ap / (1.0f + 2.0f * ap - ap * ly(x-1));

			data.m(x, y) = 0.0f;
			data.t4(x, y) = 0.0f;
			data.q4(x, y) = 0.0f;
		}
	}

	for(size_t y = 0; y < data.dimY(); y++) {
		data.t(0, y) = consts.TemOnBounds;
		data.t(data.dimX() - 1, y) = consts.TemOnBounds;
	}

	for (size_t x = 0; x < data.dimX(); x++) {
		data.t(x, 0) = consts.TemOnBounds;
		data.t(x, data.dimY() - 1) = consts.TemOnBounds;
	}
	
	//Создать функторы CPU
}

std::string FireSpreadSimpleModel::PrintData() const {
	return "";
}

void FireSpreadSimpleModel::SynchronizeWithGpu() {
	if (isGpuOn())
		_data_dev.PutTo(_data);
}

void FireSpreadSimpleModel::PrepareDataForGpu(const Sonsode::GpuDevice &gpu, size_t orderNumber) {
	_data_dev = FireSpreadDataD(gpu, _data);
	lx_dev = DeviceData1D<float>(gpu, lx);
	ly_dev = DeviceData1D<float>(gpu, ly);
	//fnGPU - создать функторы GPU ///!!!
}

void FireSpreadSimpleModel::FreeDataForGpus() {
	_data_dev.Erase();
	lx_dev.Erase();
	ly_dev.Erase();
}

void FireSpreadSimpleModel::CalculationMethod_CPU() {
	//Противоточные произодные
	for (size_t y = 1; y <= _data.dimY()-2; y++) {
		for (size_t x = 1; x <= _data.dimX()-2; x++) {
			float temX = (u.x > 0) ? (r0 * (_data.t(x, y) - _data.t(x-1, y))) : (r0 * (_data.t(x+1, y) - _data.t(x, y)));
			float temY = (u.y > 0) ? (r0 * (_data.t(x, y) - _data.t(x, y-1))) : (r0 * (_data.t(x, y+1) - _data.t(x, y)));

			float temzX = _consts.Danu * (_data.t(x+1, y) - 2.0f * _data.t(x, y) + _data.t(x-1, y)) / hh;
			float temzY = _consts.Danu * (_data.t(x, y+1) - 2.0f * _data.t(x, y) + _data.t(x, y-1)) / hh;

			_data.t4(x, y) = temzX + temzY - u.x * temX - u.y * temY;
		}
	}

	//Новые температуры
	//Прогонка по Х
	for (size_t y = 1; y < _data.dimY()-1; y++) {
		for (size_t x = 1; x < _data.dimX()-1; x++) {
			_data.q4(x, y) = -ap * _data.t(x-1, y) + (1.0f + 2.0f * ap) * _data.t(x, y) 	- ap * _data.t(x+1, y);
			_data.q4(x, y) = _data.q4(x, y) + tau() * _data.t4(x, y);
		}
	}

	for (size_t y = 1; y < _data.dimY()-1; y++) {
		for (size_t x = 1; x < _data.dimX()-1; x++) {
			_data.m(x+1, y) = (_data.q4(x, y) / ap + _data.m(x, y)) * ly(x+1);
		}
	}

	for (size_t y = 1; y < _data.dimY()-1; y++) {
		for (size_t x = 1; x < _data.dimX()-1; x++) {
			int k = (_data.dimX()-2) - (x + 1) + 3;
			_data.t(k-1, y) = ly(k) * _data.t(k, y) + _data.m(k, y);
		}
	}

	//Прогонка по У
	for (size_t x = 1; x < _data.dimX()-1; x++) {
		for (size_t y = 1; y < _data.dimY()-1; y++) {
			_data.q4(x, y) = -ap * _data.t(x, y-1) + (1.0f + 2.0f * ap) * _data.t(x, y) - ap * _data.t(x, y+1);
			_data.q4(x, y) = _data.q4(x, y) + tau() * _data.t4(x, y);
		}
	}

	for (size_t x = 1; x < _data.dimX()-1; x++) {
		for (size_t y = 1; y < _data.dimY()-1; y++) {
			_data.m(x, y+1) = (_data.q4(x, y) / ap + _data.m(x, y)) * lx(y+1);
		}
	}

	for (size_t x = 1; x < _data.dimX() - 1; x++) {
		for (size_t y = 1; y < _data.dimY() - 1; y++)	 {
			int k = (_data.dimY() - 2) - (y + 1) + 3;
			_data.t(x, k-1) = lx(k) * _data.t(x, k) + _data.m(x ,k);
		}
	}

	//Горение
	if (currentIteration() >= _consts.IterFireBeginNum) {
		for (size_t y = 0; y < _data.dimY(); y++) {
			for (size_t x = 0; x < _data.dimX(); x++) {
				float fgor = _consts.Qbig * pow(fabs(_data.t(x, y)), _consts.Mstep) * exp(-_consts.Tzv / _data.t(x, y));
				if (_data.t(x, y) < _consts.TemKr)
					fgor = 0.0f;
				_data.roFuel(x, y) = _data.roFuel(x, y) / (1.0f + tau() * fgor);
				_data.t(x, y) = _data.t(x, y) + tau() * _consts.Qlitl * _data.roFuel(x, y) * fgor 
					/ (1.0f + _consts.Ks * _consts.Humidity);
			}
		}
	}
}

void FireSpreadSimpleModel::CalculationMethod_GPU() {
	GpuOn();

	_data_dev.gpu().SetAsCurrent();

	Run_Kernel_FireSpreadSimpleModel_CounterflowDerivative(_data_dev, u, r0, _consts.Danu, hh);
	_data_dev.gpu().Synchronize();
	_data_dev.gpu().CheckLastErr();

	Run_Kernel_FireSpreadSimpleModel_RunAroundX(_data_dev, ly_dev, ap, tau());
	_data_dev.gpu().Synchronize();
	_data_dev.gpu().CheckLastErr();
	
	Run_Kernel_FireSpreadSimpleModel_RunAroundY(_data_dev, lx_dev, ap, tau());
	_data_dev.gpu().Synchronize();
	_data_dev.gpu().CheckLastErr();
	
	if (currentIteration() >= _consts.IterFireBeginNum) {
		Run_Kernel_FireSpreadSimpleModel_Fire(_data_dev, _consts.TemKr, _consts.Qbig, _consts.Mstep,
			_consts.Tzv, tau(), _consts.Qlitl, _consts.Ks, _consts.Humidity);
		_data_dev.gpu().Synchronize();
		_data_dev.gpu().CheckLastErr();
	}
}