#pragma once

#include <iostream>
#include "IterativeModel.h"
#include "ForestFireTestModel.h"
#include "Heat2DTestModel.h"
#include "Heat3DTestModel.h"
#include "AirFlowTestModel.h"
#include "OilSpillTestModel.h"

namespace InitRoutines {
	//Инициализирует и возвращает модель теплораспределения в пластине
	Heat2DTestModel* GetInitedHeat2D();
	Heat2DTestModel* GetInitedHeat2D(size_t dimX, size_t dimY);

	//Инициализирует и возвращает модель теплораспределения в объеме
	Heat3DTestModel* GetInitedHeat3D();
	Heat3DTestModel* GetInitedHeat3D(size_t dimX, size_t dimY, size_t dimZ);

	//Инициализирует и возвращает модель распространения пожара
	ForestFireTestModel* GetInitedForestFire();
	ForestFireTestModel* GetInitedForestFire(size_t dimX, size_t dimY);

	//Инициализирует и возвращает модель распределения температурных потоков
	AirFlowTestModel* GetInitedAirFlow();
	AirFlowTestModel* GetInitedAirFlow(size_t dimX, size_t dimY, size_t dimZ);

	//Инициализирует и возвращает модель разлива нефти по водной поверхности
	OilSpillTestModel* GetInitedOilSpill();
	OilSpillTestModel* GetInitedOilSpill(size_t dimX, size_t dimY);

	//Получает модель в зависимости от выбора пользователя
	IterativeModel *GetModelByUserChange();
}