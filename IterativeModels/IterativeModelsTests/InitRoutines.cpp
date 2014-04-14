#include "InitRoutines.h"

IterativeModel* InitRoutines::GetModelByUserChange() {
	//Меню - выбор модели пользователем
	int pointMenu = -1;
	std::cout << "[0] Выход\n"
			<< "[1] Модель теплопроводности в пластине\n"
			<< "[2] Модель теплопроводности в пространстве\n"
			<< "[3] Модель распространения пожара\n"
			<< "[4] Модель распределения температурных потоков\n"
			<< "[5] Модель нефтеразлива\n";
	
	while (pointMenu < 0 || pointMenu > 9)
		std::cin >> pointMenu;

	switch (pointMenu) {
		case 0://Выход из приложения
			return 0;
		case 1://Модель теплораспределения
			return GetInitedHeat2D();
		case 2://Модель теплораспределения 3D
			return GetInitedHeat3D();
		case 3://Модель распространения пожара
			return GetInitedForestFire();
		case 4://Модель распределения температурных потоков
			return GetInitedAirFlow();
		case 5://Модель нефтеразлива
			return GetInitedOilSpill();
	}

	return 0;
}

Heat2DTestModel* InitRoutines::GetInitedHeat2D() {
	return GetInitedHeat2D(10, 10);
}

Heat2DTestModel* InitRoutines::GetInitedHeat2D(size_t dimX, size_t dimY) {
	auto t = Sonsode::HostData2D<float>(dimX, dimY);

	for (size_t x = 0; x < t.dimX(); x++) {
		for (size_t y = 0; y < t.dimY(); y++) {
			if (x == 0)
				t(x, y) = 100.0f;
			else if (y == 0)
				t(x, y) = 200.0f;
			else
				t(x, y) = 0.0f;
		}
	}

	float h = 0.5f;
	float a = 0.3f;
	float tau = 0.1f;

	return new Heat2DTestModel(t, h, a, tau);
}

Heat3DTestModel* InitRoutines::GetInitedHeat3D() {
	return GetInitedHeat3D(30, 30, 30);
}

Heat3DTestModel* InitRoutines::GetInitedHeat3D(size_t dimX, size_t dimY, size_t dimZ) {
	auto t = Sonsode::HostData3D<float>(dimX, dimY, dimZ);

	for (size_t x = 0; x < t.dimX(); x++) {
		for (size_t y = 0; y < t.dimY(); y++) {
			for (size_t z = 0; z < t.dimZ(); z++) {
				if (x == 0)
					t(x, y, z) = 100.0f;
				else if (x == t.dimX() - 1)
					t(x, y, z) = 200.0f;
				else if (y == 0)
					t(x, y, z) = 300.0f;
				else if (y == t.dimY() - 1)
					t(x, y, z) = 400.0f;
				else if (z == 0)
					t(x, y, z) = 500.0f;
				else if (z == t.dimZ() - 1)
					t(x, y, z) = 600.0f;
				else
					t(x, y, z) = 0.0f;
			}
		}
	}

	float h = 0.5f;
	float a = 0.3f;
	float tau = 0.1f;

	auto heatConductivity3DModel = new Heat3DTestModel(t, h, a, tau);
	heatConductivity3DModel->curPlane = Vector3D<size_t>(t.dimX() / 2, t.dimY() / 2, t.dimZ() / 2);

	return heatConductivity3DModel;
}

ForestFireTestModel* InitRoutines::GetInitedForestFire() {
	return GetInitedForestFire(50, 50);
}

ForestFireTestModel* InitRoutines::GetInitedForestFire(size_t dimX, size_t dimY) {
	
	float h = 45.0f;
	float tau = 0.1f;
	float humidity = 0.3f;
	float windAngle = 3.14f;
	float windSpeed = 5.0f;
	float m2 = 0.001f;
	float danu = 10.0f;
	float temOnBounds = 0.0f;
	int iterFireBeginNum = 0;
	float qbig = 1.0f;
	int mstep = 1;
	float tzv = 10.0f;
	float temKr = 3.0f;
	float qlitl = 300.0f;
	float ks = 0.01f;
	float enviromentTemperature = 0.0f;

	ForestFireConsts consts(h, tau, humidity, windAngle, windSpeed, m2, danu, temOnBounds,
													iterFireBeginNum, qbig, mstep, tzv, temKr, qlitl, ks, enviromentTemperature);
		
	ForestFireDataH data(dimX, dimY);
	data.Fill(enviromentTemperature);

	for(size_t x = 0; x < data.dimX(); x++) {
		for(size_t y = 0; y < data.dimY(); y++) {
			if (x == data.dimX() / 2 && y == data.dimY() / 2)
				data.t(x, y) = 400.0f;

			data.roFuel(x, y) = 25.0f;
		}
	}

	return new ForestFireTestModel(consts, data);
}

AirFlowTestModel* InitRoutines::GetInitedAirFlow() {
	return GetInitedAirFlow(20, 20, 20);
}

AirFlowTestModel* InitRoutines::GetInitedAirFlow(size_t dimX, size_t dimY, size_t dimZ) {
	float defT = 15.0f + 273.0f;
	float startRo = 1.3f;
	
	AirFlowDataH data(dimX, dimY, dimZ);
	for (size_t x = 0; x < dimX; x++) {
		for (size_t y = 0; y < dimY; y++) {
			for (size_t z = 0; z < dimZ; z++) {
				data.ux(x, y, z) = 0.0f;
				data.uy(x, y, z) = 0.0f;
				data.uz(x, y, z) = 0.0f;
				data.ro(x, y, z) = startRo;
				data.t(x, y, z) = defT;

				if ((x == 0 || x == dimX - 1) && y > 5 && y <= 10 && z > 5 && z <= 10)
					data.ux(x, y, z) = 20.0f;

				if ((x == 0 || x == dimX - 1) && 
					(((y == 6 || y == 10) && (z > 5 && z <= 10)) || 
					((z == 6 || z == 10) && (y > 5 && y <= 10))))
					data.ux(x, y, z) = 5.0f;
			}
		}
	}

	AirFlowConsts consts;
	consts.H = 0.1f;
	consts.Tau = 0.001f;
	consts.Nu = 0.5f;
	consts.D = 0.2f;
	
	auto airFlowDistributionModel = new AirFlowTestModel(consts, data);
	airFlowDistributionModel->curPlane = Vector3D<size_t>(data.dimX() / 2, data.dimY() / 2, data.dimZ() / 2);

	return airFlowDistributionModel;
}

OilSpillTestModel* InitRoutines::GetInitedOilSpill() {
	return GetInitedOilSpill(100, 100);
}

OilSpillTestModel* InitRoutines::GetInitedOilSpill(size_t dimX, size_t dimY) {
	float defWaterUx = 0.0f;
	float defWaterUy = 0.0f;
	float defOilUx = 0.0f;
	float defOilUy = 0.0f;
	float defW = 1.0f;
	float defDeep = 10.0f;
	float defImpurity = 3.0f;
	float defPress = 0.0f;

	OilSpillDataH data(dimX, dimY);
	for (size_t x = 0; x <= dimX - 1; x++) {
		for (size_t y = 0; y <= dimY - 1; y++) {
			data.waterUx(x, y) = defWaterUx;
			data.waterUy(x, y) = defWaterUy;
			data.oilUx(x, y) = defOilUx;
			data.oilUy(x, y) = defOilUy;
			data.w(x, y) = defW;
			data.deep(x, y) = defDeep;
			data.impurity(x, y) = defImpurity;
			data.press(x, y) = defPress;
		}
	}

	OilSpillConsts consts;
	consts.Temperature = 20.0f;
	consts.Tau = 0.3f;
	consts.CoriolisFactor = 0.001f;
	consts.H = 200.0f;
	consts.Mult = 10.0f;
	consts.BackImpurity = 3.0f;
	consts.Ukl = Vector2D<float>(0.0f, 0.0f);
	consts.Sopr = 0.001f;
	consts.Beta = Vector2D<float>(0.0001f, 0.0001f);
	consts.G = 9.81f;
	consts.Tok = 0.001f;
	consts.WindSpeed = Vector2D<float>(30.0f, 30.0f);

	for(size_t x = 0; x < dimX; x++) {
		for(size_t y = 0; y < dimY; y++) {
			if (x == 0) {
				data.waterUx(x, y) = 0.3f;
				data.oilUx(x, y) = 0.3f;
			}

			if (x == dimX - 1) {
				data.waterUx(x, y) = 1.0f;
			}

			if (y == 0) {
				data.waterUy(x, y) = 0.25f;
				data.oilUy(x, y) = 0.25f;
			}
			
			if (50 <= x && x <= 52 && 50 <= y && y <= 52) {
				data.impurity(x, y)	= 500.0f;
			}

			if (58 <= x && x <= 60 && 58 <= y && y <= 60) {
				data.deep(x, y) = 0.0f;
			}
		}
	}

	return new OilSpillTestModel(consts, data);
}