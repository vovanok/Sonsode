#include <iostream>
#include <string>
#include <Windows.h>

#include "GraphicMgr.h"
#include "IterativeModel.h"
#include "InitRoutines.h"
#include "Tests.h"
#include "CapacityTests.h"

//#include <algorithm>
#include <ctime>
#include <sstream>
#include "Geometry.hpp"
#include "Region.h"
#include "KmlMgr.h"
#include "Vectorizator.h"
#include "DataVisualizationException.h"
#include "FireKmlVisualizator.h"
#include "OilKmlVisualizator.h"

IterativeModel *model;
GraphicMgr* grEngine = 0;
void CloseApplication();
void StartUiApp(int argc, char ** argv);
void NextModelIteration(std::string methodName);
void StartStopProcessModelIteration(std::string methodName);
void ForestFireKml(int argc, char ** argv);
void OilSpillageKml(int argc, char ** argv);

void main(int argc, char ** argv) {
	setlocale(LC_ALL, "rus");

	////Heat conductivity 2D test
	//CapacityTests::HeatConductivity2D(10, 100, 2000, 100, "cpu_gaussseidel");
	//CapacityTests::HeatConductivity2D(10, 100, 2000, 100, "cpu_sweep");
	//CapacityTests::HeatConductivity2D(10, 100, 2000, 100, "gpu_gaussseidel_direct");
	//CapacityTests::HeatConductivity2D(10, 100, 2000, 100, "gpu_gaussseidel_chess");
	//CapacityTests::HeatConductivity2D(10, 100, 2000, 100, "gpu_gaussseidel_withoutconflicts");
	//CapacityTests::HeatConducivity2D(10, 100, 2000, 100, "gpu_gaussseidel_direct_overlay");
	//CapacityTests::HeatConductivity2D(10, 100, 2000, 100, "gpu_sweep_linedevide");
	//CapacityTests::HeatConductivity2D(10, 100, 2000, 100, "gpu_sweep_blockdevide");

	//Heat conductivity 3D test
	//CapacityTests::HeatConductivity3D(10, 10, 400, 10, "cpu_gaussseidel");
	//CapacityTests::HeatConductivity3D(10, 10, 400, 10, "gpu_gaussseidel_direct");
	//CapacityTests::HeatConductivity3D(10, 10, 400, 10, "cpu_sweep");
	//CapacityTests::HeatConductivity3D(10, 10, 400, 10, "gpu_sweep_linedevide");

	//CapacityTests::OilSpillage(10, 100, 1400, 100, "cpu");
	//CapacityTests::OilSpillage(10, 100, 2000, 100, "gpu");

	//CapacityTests::FireSpreadRealAndModelTimeRelation(3600, 600, 500, 500, 100, "cpu");
	//CapacityTests::FireSpreadRealAndModelTimeRelation(3600, 600, 400, 500, 100, "gpu");
	//CapacityTests::OilSpillageRealAndModelTimeRelation(3600, 600, 100, 400, 100, "gpu");
	//CapacityTests::OilSpillageRealAndModelTimeRelation(3600, 600, 300, 400, 100, "cpu");

	//CapacityTests::AirFlow(10, 50, 190, 10, "cpu");
	//CapacityTests::AirFlow(10, 150, 190, 10, "gpu");

	StartUiApp(argc, argv);

	//Working with KML
	//ForestFireKml(argc, argv);
	//OilSpillageKml(argc, argv);
	
	std::cout << "Нажмите любую клавишу для продолжения..." << std::endl;
	std::cin.get();

	CloseApplication();
}

#pragma region Actions

void Action_Nothing() { }
void Action_Close() { CloseApplication(); }
void Action_PrintModelData() { model->PrintData(); }
void Action_Up() { grEngine->CameraRotate(CameraRotateDirection::Up); }
void Action_Down() { grEngine->CameraRotate(CameraRotateDirection::Down); }
void Action_Left() { grEngine->CameraRotate(CameraRotateDirection::Left); }
void Action_Right() { grEngine->CameraRotate(CameraRotateDirection::Right); }
void Action_ZoomIn() { grEngine->CameraZoom(CameraZoomDirection::In); }
void Action_ZoomOut() {	grEngine->CameraZoom(CameraZoomDirection::Out); }

#pragma endregion

#pragma region Application routines

void NextModelIteration(std::string methodName) {
	if (model == nullptr)
		return;

	try {
		double time;
		size_t iterNumber;
		model->NextIteration(methodName, time, iterNumber);

		std::cout << methodName << ": " << iterNumber << ", время: " << time << std::endl;
	} catch(std::string e) {
		std::cout << e << std::endl;
	}
}

void StartStopProcessModelIteration(std::string methodName) {
	if (!grEngine->IsRunAnimation()) {
		grEngine->RegisterPreDrawHandler([=]() { NextModelIteration(methodName); } );
		grEngine->StartAnimation();
	} else {
		grEngine->RegisterPreDrawHandler(Action_Nothing);
		grEngine->StopAnimation();
	}
}

void CloseApplication() {
	GraphicMgr::Free();
	delete model;
	exit(0);
}

BOOL WINAPI ConsoleHandler(DWORD CEvent) {
    switch(CEvent) {
		//case CTRL_C_EVENT: 
		//	logMsg << "CTRL_C_EVENT\n";
		//	break;
		//case CTRL_BREAK_EVENT:
		//	logMsg << "CTRL_BREAK_EVENT\n";
		//	break;
		//case CTRL_LOGOFF_EVENT:
		//	logMsg << "CTRL_LOGOFF_EVENT\n";
		//	break;
		////case CTRL_SHUTDOWN_EVENT: break;
		//case CTRL_SHUTDOWN_EVENT:
		//	logMsg << "CTRL_SHUTDOWN_EVENT\n";
		//	break;
		case CTRL_CLOSE_EVENT:
			CloseApplication();
			break;
    }
    return TRUE;
}

void GetButtonForManageIteration(
		size_t methodOrderNum, std::string& singleIterationButton, std::string& processIterationsButton) {

	switch(methodOrderNum) {
		case 0:
			singleIterationButton = "1";
			processIterationsButton = "ZzЯя";
			break;
		case 1:
			singleIterationButton = "2";
			processIterationsButton = "XxЧч";
			break;
		case 2:
			singleIterationButton = "3";
			processIterationsButton = "CcСс";
			break;
		case 3:
			singleIterationButton = "4";
			processIterationsButton = "VvМм";
			break;
		case 4:
			singleIterationButton = "5";
			processIterationsButton = "BbИи";
			break;
		case 5:
			singleIterationButton = "6";
			processIterationsButton = "NnТт";
			break;
		case 6:
			singleIterationButton = "7";
			processIterationsButton = "MmЬь";
			break;
		case 7:
			singleIterationButton = "8";
			processIterationsButton = "<,Бб";
			break;
		case 8:
			singleIterationButton = "9";
			processIterationsButton = ">.Юю";
			break;
		default:
			singleIterationButton = "";
			processIterationsButton = "";
			break;
	}
}

void StartUiApp(int argc, char ** argv) {
	if (SetConsoleCtrlHandler((PHANDLER_ROUTINE)ConsoleHandler, TRUE) == FALSE)
		CloseApplication();

	try {
		model = InitRoutines::GetModelByUserChange();
		if (model == 0)
			CloseApplication();
	} catch (std::string e) {
		std::cout << "Ошибка получения прототипа модели: " << e << std::endl;
		return;
	}
	
	grEngine = GraphicMgr::New(argc, argv, "Iterative models", false, false);
	grEngine->AddPresentObj(dynamic_cast<IPresentable *>(model));

	grEngine->AddUpdateKeyboardHandler('0', Action_Close);
	std::cout	<< "0 - закрыть приложение" << std::endl;

	int curNum = 0;
	for (auto calcMethod : model->CalculationMethods) {
		std::string singleButton;
		std::string processButton;

		GetButtonForManageIteration(curNum, singleButton, processButton);
		if (singleButton == "" || processButton == "")
			continue;

		std::string methodName = calcMethod.first;

		grEngine->AddUpdateKeyboardHandler(singleButton, [=]() { NextModelIteration(methodName); });
		grEngine->AddUpdateKeyboardHandler(processButton, [=]() { StartStopProcessModelIteration(methodName); });

		std::cout << singleButton << " - одна итерация; " << processButton << " - старт/стоп. Метод " << methodName << std::endl;
		curNum++;
	}

	grEngine->AddUpdateKeyboardHandler("PpЗз", Action_PrintModelData);
	std::cout << "P - печать данных модели" << std::endl;

	grEngine->AddUpdateKeyboardHandler("WwЦц", Action_Up);
	grEngine->AddUpdateKeyboardHandler("AaФф", Action_Left);
	grEngine->AddUpdateKeyboardHandler("SsЫы", Action_Down);
	grEngine->AddUpdateKeyboardHandler("DdВв", Action_Right);
	std::cout << "W, A, S, D - поворот камеры" << std::endl;

	grEngine->AddUpdateKeyboardHandler("RrКк", Action_ZoomIn);
	grEngine->AddUpdateKeyboardHandler("FfАа", Action_ZoomOut);
	std::cout << "R, F - приблизить/отдалить" << std::endl;

	grEngine->AddUpdateKeyboardHandler("JjОоGgПпHhРрYyНнKkЛлIiШшUuГгTtЕе", Action_Nothing);
	std::cout << "J, G, H, Y, K, I - движение плоскостей 3D среза" << std::endl;
	std::cout << "U - переключение типа полей" << std::endl;
	std::cout << "T - 3D / срезы" << std::endl;

	grEngine->Run();
}

#pragma endregion

#pragma region Kml modelers

#pragma region Common

const float EARTH_RADIUS = 6371.0f;
const float PI = 3.1415926535897932384626433832795f;
const float ABSOLUTE_TEMPERATURE_STRIDE = 273.0f;
const int SECONDS_IN_HOUR = 3600;

float CelsiusToKelvin(float celsiusValue) {
	return celsiusValue + ABSOLUTE_TEMPERATURE_STRIDE;
}

float HoursToSeconds(float hours) {
	return hours * (float)SECONDS_IN_HOUR;
}

Geometry::Rect<float> GetRegionsClearence(const std::vector<Region>& regions) {
	Geometry::Rect<float> result = regions[0].GetClearanceBorders();

	Geometry::Rect<float> curClearence;
	for (Region region : regions) {
		curClearence = region.GetClearanceBorders();
		
		result.point1.x = min(curClearence.point1.x, result.point1.x);
		result.point1.y = min(curClearence.point1.y, result.point1.y);
		result.point2.x = max(curClearence.point2.x, result.point2.x);
		result.point2.y = max(curClearence.point2.y, result.point2.y);
	}

	return result;
}

float LattitudeCloserEquator(float latitude1, float latitude2) {
	if (abs(latitude1) < abs(latitude2))
		return latitude1;
	return latitude2;
}

Geometry::Rect<float> MeterClearenceByGradusArea(const Geometry::Rect<float>& gradusArea) {
	Geometry::Rect<float> result(Geometry::Point<float>(0.0f, 0.0f), Geometry::Point<float>(0.0f, 0.0f));

	float earthC = 2.0f * PI * EARTH_RADIUS;
	Geometry::Point<float> oneGradInKm(
		1000.0f * (earthC * cosf(LattitudeCloserEquator(gradusArea.point1.y, gradusArea.point2.y) * (PI / 180.0f))) / 360.0f,
		1000.0f * (earthC / 360.0f));

	result.point2.x = abs(gradusArea.point1.x - gradusArea.point2.x) * oneGradInKm.x;// * 2.0f;
	result.point2.y = abs(gradusArea.point1.y - gradusArea.point2.y) * oneGradInKm.y;// * 2.0f;

	return result;
}

void DiscretizeRegions(const std::vector<Region>& regions, const Geometry::Point<float>& luPoint,
											 float h, HostData2D<bool>& discreteField) {
	discreteField.Fill(false);
	for (auto region : regions) {
		Vectorizator::DiscretizeAndAppend(region.outerBounder, luPoint, h, true, discreteField);
		for (auto innerBounder : region.innerBounders)
			Vectorizator::DiscretizeAndAppend(innerBounder, luPoint, h, false, discreteField);
	}
}

void NormalizeRegions(std::vector<Region>& regions, const Geometry::Rect<float>& src,
											const Geometry::Rect<float>& dst) {
	for (auto& region : regions)
		region.Normalize(src, dst);
}

std::string ForecastKmlFileName() {
	std::stringstream ss;
	time_t t = time(0);
	tm* now = localtime(&t);

	ss << "forecast_" 
		<< now->tm_mday << "_"
		<< now->tm_mon + 1 << "_"
		<< now->tm_year + 1900 << "_"
		<< now->tm_hour << "_"
		<< now->tm_min << "_"
		<< now->tm_sec << ".kml";

	return ss.str();
}

#pragma endregion

#pragma region Forest fires
std::vector<Region> GetForestFireForecast(const FireSpreadConsts& consts,
																					std::vector<Region> forestRegions, std::vector<Region> fireRegions,
																					float fireplaceTemperatureK, float forestRoFuel, float nonForestRoFuel,
																					float minFireTemperatureK, float modelingTimeSec, bool isUseGpu) {
	//Получение габаритов леса градусах и метрах
	Geometry::Rect<float> gradForestClearence = GetRegionsClearence(forestRegions);
	Geometry::Rect<float> meterForestClearence = MeterClearenceByGradusArea(gradForestClearence);

	//Нормализация координат между загруженными из файлов и более приемлемыми для расчетов
	NormalizeRegions(forestRegions, gradForestClearence, meterForestClearence);
	NormalizeRegions(fireRegions, gradForestClearence, meterForestClearence);
	
	size_t dimX = (size_t)((meterForestClearence.point2.x - meterForestClearence.point1.x) / consts.H);
	size_t dimY = (size_t)((meterForestClearence.point2.y - meterForestClearence.point1.y) / consts.H);

	std::cout << "dimX = " << dimX << "; dimY = " << dimY << std::endl;

	FireSpreadDataH data(dimX, dimY);
	HostData2D<bool> forestDiscreteGrid(dimX, dimY), fireDiscreteGrid(dimX, dimY);
	DiscretizeRegions(forestRegions, meterForestClearence.point1, consts.H, forestDiscreteGrid);
	DiscretizeRegions(fireRegions, meterForestClearence.point1, consts.H, fireDiscreteGrid);

	for (size_t x = 0; x < data.dimX(); x++) {
		for (size_t y = 0; y < data.dimY(); y++) {
			data.roFuel(x, y) = forestDiscreteGrid(x, y) ? forestRoFuel : nonForestRoFuel;
			data.t(x, y) = fireDiscreteGrid(x, y) ? fireplaceTemperatureK : consts.EnviromentTemperature;
		}
	}

	model = new FireSpreadSimpleModel(consts, data);
	size_t countIteration = modelingTimeSec / consts.Tau;
	for (size_t iterNum = 0; iterNum < countIteration; iterNum++)
		model->NextIteration(isUseGpu ? "gpu" : "cpu");

	model->SynchronizeWithGpu();

	for (size_t x = 0; x < data.dimX(); x++)
		for (size_t y = 0; y < data.dimY(); y++)
			fireDiscreteGrid(x, y) = (data.t(x, y) >= minFireTemperatureK);

	auto forecastPolygons = Vectorizator::Vectorize(fireDiscreteGrid, meterForestClearence.point1, consts.H);

	std::vector<Region> forecastRegions;
	for (auto fireForecastPolygon : forecastPolygons)
		forecastRegions.push_back(Region(fireForecastPolygon, std::vector<Geometry::Polygon<float>>(0)));

	//Нормализация координат в обратную сторону
	for (Region& forecastRegion : forecastRegions)
		forecastRegion.Normalize(meterForestClearence, gradForestClearence);

	forestDiscreteGrid.Erase();
	fireDiscreteGrid.Erase();
	
	return forecastRegions;
}

void ForestFireKml(int argc, char ** argv) {
	try {
		const float WINDSPEED_INHIBITION_FACTOR = 0.001f;

		const float fireplaceTemperature = CelsiusToKelvin(700.0f);
		const float minFireTemperature = CelsiusToKelvin(50.0f);
		const float forestRoFuel = 10.0f;
		const float nonForestRoFuel = 0.0f;
		const float modelingTime = HoursToSeconds(1.0f);
		const bool isUseGpu = false;
		
		//Загрузка регионов пожара и леса из файлов
		std::cout << "Загрузка файлов..." << std::endl;
		std::vector<Region> forestRegions = KmlMgr::LoadPolygonsFromFile("forest.kml");
		std::vector<Region> fireRegions = KmlMgr::LoadPolygonsFromFile("fire.kml");

		FireSpreadConsts consts;
		consts.H = 40.0f;
		consts.Tau = 20.0f;
		consts.Humidity = 0.3f;
		consts.WindAngle = 0.0f;
		consts.WindSpeed = 0.0f * WINDSPEED_INHIBITION_FACTOR;
		consts.M2 = 0.001f;
		consts.Danu = 1.0f;
		consts.TemOnBounds = CelsiusToKelvin(10.0f);
		consts.IterFireBeginNum = 0;
		consts.Qbig = 1.0f;
		consts.Mstep = 0.5f;
		consts.Tzv = CelsiusToKelvin(100.0f);
		consts.TemKr = CelsiusToKelvin(100.0f);
		consts.Qlitl = 50.0f;
		consts.Ks = 0.01f;
		consts.EnviromentTemperature = CelsiusToKelvin(0.6f);

		std::cout << "Расчет прогноза..." << std::endl;
		std::vector<Region> forecastRegions
			= GetForestFireForecast(consts, forestRegions, fireRegions, fireplaceTemperature, forestRoFuel,
															nonForestRoFuel, minFireTemperature, modelingTime, isUseGpu);

		std::cout << "Сохранение файла..." << std::endl;
		KmlMgr::SavePolygonsToFile(ForecastKmlFileName(), "forecastTemplate.kml", forecastRegions);

#pragma region Fire KML visualization
		Geometry::Rect<float> forestClearence = GetRegionsClearence(forestRegions);
		FireKmlVisualizator visualizator(consts.H, forestClearence,
			Geometry::Rect<float>(Geometry::Point<float>(-30.0f, -30.0f), Geometry::Point<float>(30.0f, 30.0f)));
		visualizator.fireRegions = fireRegions;
		visualizator.forestRegions = forestRegions;
		visualizator.forecastRegions = forecastRegions;

		grEngine = GraphicMgr::New(argc, argv, "Iterative models", false, false);
		grEngine->AddPresentObj(dynamic_cast<IPresentable *>(&visualizator));

		grEngine->AddUpdateKeyboardHandler('0', Action_Close);
		std::cout	<< "0 - закрыть приложение" << std::endl;

		grEngine->Run();
#pragma endregion

	} catch (DataVisualizationException e) {
		std::cout << "Ошибка в модуле визуализации: " << e.what() << std::endl;
	} catch (std::string e) {
		std::cout << "Ошибка в вычислениях: "	<< e << std::endl;
	} catch (std::exception e) {
		std::cout << "Неизвестная ошибка: " << e.what() << std::endl;
	}
}
#pragma endregion

#pragma region Oil spillage
std::vector<Region> GetOilSpillageForecast(const OilSpillageConsts& consts,
																					std::vector<Region> waterRegions, std::vector<Region> oilRegions,
																					float oilplaceImpurity, float waterDeep, float nonWaterDeep,
																					float minImpurity, float modelingTimeSec, bool isUseGpu,
																					HostData2D<bool>& waterDiscreteGrid, HostData2D<bool>& oilDiscreteGrid,
																					HostData2D<bool>& forecastDiscreteGrid,
																					Geometry::Rect<float>& meterClearence) {
	//Получение габаритов водоема в градусах и метрах
	Geometry::Rect<float> gradWaterClearence = GetRegionsClearence(waterRegions);
	Geometry::Rect<float> meterWaterClearence = MeterClearenceByGradusArea(gradWaterClearence);

	//!!!
	meterClearence = meterWaterClearence;

	//Нормализация координат между загруженными из файлов и более приемлемыми для расчетов
	NormalizeRegions(waterRegions, gradWaterClearence, meterWaterClearence);
	NormalizeRegions(oilRegions, gradWaterClearence, meterWaterClearence);

	size_t dimX = (size_t)((meterWaterClearence.point2.x - meterWaterClearence.point1.x) / consts.H);
	size_t dimY = (size_t)((meterWaterClearence.point2.y - meterWaterClearence.point1.y) / consts.H);

	std::cout << "dimX = " << dimX << "; dimY = " << dimY << std::endl;

	OilSpillageDataH data(dimX, dimY);
	//!!!
	//HostData2D<bool> waterDiscreteGrid(dimX, dimY), oilDiscreteGrid(dimX, dimY);
	waterDiscreteGrid = HostData2D<bool>(dimX, dimY);
	oilDiscreteGrid = HostData2D<bool>(dimX, dimY);
	DiscretizeRegions(waterRegions, meterWaterClearence.point1, consts.H, waterDiscreteGrid);
	DiscretizeRegions(oilRegions, meterWaterClearence.point1, consts.H, oilDiscreteGrid);

	for (size_t x = 0; x < data.dimX(); x++) {
		for (size_t y = 0; y < data.dimY(); y++) {
			data.deep(x, y) = waterDiscreteGrid(x, y) ? waterDeep : nonWaterDeep;
			data.impurity(x, y) = oilDiscreteGrid(x, y) ? oilplaceImpurity : consts.BackImpurity;
			data.w(x, y) = 1.0f;
			data.waterUx(x, y) = 0.0f;
			data.waterUy(x, y) = 0.0f;
			data.oilUx(x, y) = 0.0f;
			data.oilUy(x, y) = 0.0f;
			data.press(x, y) = 0.0f;
		}
	}

	model = new OilSpillageImprovedModel(consts, data);
	size_t countIteration = modelingTimeSec / consts.Tau;
	for (size_t iterNum = 0; iterNum < countIteration; iterNum++)
		model->NextIteration(isUseGpu ? "gpu" : "cpu");

	model->SynchronizeWithGpu();

	forecastDiscreteGrid = HostData2D<bool>(dimX, dimY);//!!!
	for (size_t x = 0; x < data.dimX(); x++)
		for (size_t y = 0; y < data.dimY(); y++)
			//oilDiscreteGrid(x, y) = (data.impurity(x, y) >= minImpurity);
			forecastDiscreteGrid(x, y) = (data.impurity(x, y) >= minImpurity);

	auto forecastPolygons = Vectorizator::Vectorize(forecastDiscreteGrid //oilDiscreteGrid
		, meterWaterClearence.point1, consts.H);

	std::vector<Region> forecastRegions;
	for (auto oilForecastPolygon : forecastPolygons)
		forecastRegions.push_back(Region(oilForecastPolygon, std::vector<Geometry::Polygon<float>>(0)));

	//Нормализация координат в обратную сторону
	for (Region& forecastRegion : forecastRegions)
		forecastRegion.Normalize(meterWaterClearence, gradWaterClearence);

	//!!!
	/*waterDiscreteGrid.Erase();
	oilDiscreteGrid.Erase();*/

	return forecastRegions;
}

void OilSpillageKml(int argc, char ** argv) {
	try {
		const float oilplaceImpurity = 200.0f;
		const float minOilImpurity = 30.0f;//20.0f;
		const float waterDeep = 10.0f;
		const float nonWaterDeep = 0.0f;
		const float modelingTime = HoursToSeconds(0.02f);//0.02f);//1.0f);
		const bool isUseGpu = false;
		
		//Загрузка регионов пожара и леса из файлов
		std::cout << "Загрузка файлов..." << std::endl;
		std::vector<Region> waterRegions = KmlMgr::LoadPolygonsFromFile("water.kml");
		std::vector<Region> oilRegions = KmlMgr::LoadPolygonsFromFile("oil.kml");

		OilSpillageConsts consts;
		consts.H = 80.0f;//40.0f;
		consts.Tau = 0.3f;//0.5f;

		consts.Ukl = Vector2D<float>(0.0f, 0.0f);
		consts.Beta = Vector2D<float>(0.0001f, 0.0001f);
		consts.WindSpeed = Vector2D<float>(0.0f, 0.0f);//2.0f, 0.5f);
		consts.Temperature = 0.0f;//20.0f;
		consts.BackImpurity = 3.0f;
		consts.CoriolisFactor = 0.001f;
		consts.Mult = 10.0f;
		consts.Sopr = 0.001f;
		consts.G = 9.81f;
		consts.Tok = 0.001f;

		std::cout << "Расчет прогноза..." << std::endl;
		HostData2D<bool> isWaterField, isOilField, isForecastField;
		Geometry::Rect<float> meterClearence;

		std::vector<Region> forecastRegions
			= GetOilSpillageForecast(consts, waterRegions, oilRegions, oilplaceImpurity, waterDeep,
															 nonWaterDeep, minOilImpurity, modelingTime, isUseGpu,
															 isWaterField, isOilField, isForecastField, meterClearence); //!!!

		std::cout << "Сохранение файла..." << std::endl;
		KmlMgr::SavePolygonsToFile(ForecastKmlFileName(), "forecastTemplate.kml", forecastRegions);

#pragma region Oil KML visualization
		Geometry::Rect<float> waterClearence = GetRegionsClearence(waterRegions);

		Geometry::Point<float> h(
			fabsf(waterClearence.point1.x - waterClearence.point2.x) / (float)isWaterField.dimX(),
			fabsf(waterClearence.point1.y - waterClearence.point2.y) / (float)isWaterField.dimY());

		OilKmlVisualizator visualizator(h, waterClearence,
			Geometry::Rect<float>(Geometry::Point<float>(-30.0f, -30.0f), Geometry::Point<float>(30.0f, 30.0f)));
		visualizator.waterRegions = waterRegions;
		visualizator.oilRegions = oilRegions;
		visualizator.forecastRegions = forecastRegions;

		visualizator.isWaterField = isWaterField;
		visualizator.isOilField = isOilField;
		visualizator.isForecastField = isForecastField;

		grEngine = GraphicMgr::New(argc, argv, "Iterative models", false, false);
		grEngine->AddPresentObj(dynamic_cast<IPresentable *>(&visualizator));

		grEngine->AddUpdateKeyboardHandler('0', Action_Close);
		std::cout	<< "0 - закрыть приложение" << std::endl;

		grEngine->Run();
#pragma endregion

	} catch (DataVisualizationException e) {
		std::cout << "Ошибка в модуле визуализации: " << e.what() << std::endl;
	} catch (std::string e) {
		std::cout << "Ошибка в вычислениях: "	<< e << std::endl;
	} catch (std::exception e) {
		std::cout << "Неизвестная ошибка: " << e.what() << std::endl;
	}
}
#pragma endregion

#pragma endregion