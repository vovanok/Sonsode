#pragma once

#include <iostream>
#include "IterativeModel.h"
#include "ForestFireTestModel.h"
#include "Heat2DTestModel.h"
#include "Heat3DTestModel.h"
#include "AirFlowTestModel.h"
#include "OilSpillTestModel.h"

namespace InitRoutines {
	//�������������� � ���������� ������ ������������������ � ��������
	Heat2DTestModel* GetInitedHeat2D();
	Heat2DTestModel* GetInitedHeat2D(size_t dimX, size_t dimY);

	//�������������� � ���������� ������ ������������������ � ������
	Heat3DTestModel* GetInitedHeat3D();
	Heat3DTestModel* GetInitedHeat3D(size_t dimX, size_t dimY, size_t dimZ);

	//�������������� � ���������� ������ ��������������� ������
	ForestFireTestModel* GetInitedForestFire();
	ForestFireTestModel* GetInitedForestFire(size_t dimX, size_t dimY);

	//�������������� � ���������� ������ ������������� ������������� �������
	AirFlowTestModel* GetInitedAirFlow();
	AirFlowTestModel* GetInitedAirFlow(size_t dimX, size_t dimY, size_t dimZ);

	//�������������� � ���������� ������ ������� ����� �� ������ �����������
	OilSpillTestModel* GetInitedOilSpill();
	OilSpillTestModel* GetInitedOilSpill(size_t dimX, size_t dimY);

	//�������� ������ � ����������� �� ������ ������������
	IterativeModel *GetModelByUserChange();
}