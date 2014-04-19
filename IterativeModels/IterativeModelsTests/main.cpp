#include <iostream>
#include <string>
#include <ctime>
#include <sstream>
#include <Windows.h>
#include "GraphicMgr.h"
#include "IterativeModel.h"
#include "InitRoutines.h"
#include "CapacityTests.h"
#include "Geometry.hpp"
#include "Region.h"
#include "KmlMgr.h"
#include "Vectorizator.h"
#include "DataVisualizationException.h"
#include "FireKmlVisualizator.h"
#include "OilKmlVisualizator.h"

using namespace DataVisualization;
using namespace DataVisualization::Graphic;
using namespace DataVisualization::Geometry;
using namespace OilSpill;
using namespace ForestFire;

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

	//#pragma region Heat2D perfomance

	//std::cout << "Heat conductivity 2D model perfomance test" << std::endl;

	//CapacityTests::ModelPerfomanceTest2D(
	//	[](int dimX, int dimY) -> IterativeModel* { return InitRoutines::GetInitedHeat2D(dimX, dimY); },
	//	"cpu_gaussseidel", 10, 100, 2000, 100);

	//CapacityTests::ModelPerfomanceTest2D(
	//	[](int dimX, int dimY) -> IterativeModel* { return InitRoutines::GetInitedHeat2D(dimX, dimY); },
	//	"cpu_sweep", 10, 100, 2000, 100);

	//CapacityTests::ModelPerfomanceTest2D(
	//	[](int dimX, int dimY) -> IterativeModel* { return InitRoutines::GetInitedHeat2D(dimX, dimY); },
	//	"gpu_gaussseidel_direct", 10, 100, 2000, 100);

	//CapacityTests::ModelPerfomanceTest2D(
	//	[](int dimX, int dimY) -> IterativeModel* { return InitRoutines::GetInitedHeat2D(dimX, dimY); },
	//	"gpu_gaussseidel_chess", 10, 100, 2000, 100);

	//CapacityTests::ModelPerfomanceTest2D(
	//	[](int dimX, int dimY) -> IterativeModel* { return InitRoutines::GetInitedHeat2D(dimX, dimY); },
	//	"gpu_gaussseidel_withoutconflicts", 10, 100, 2000, 100);

	//CapacityTests::ModelPerfomanceTest2D(
	//	[](int dimX, int dimY) -> IterativeModel* { return InitRoutines::GetInitedHeat2D(dimX, dimY); },
	//	"gpu_gaussseidel_direct_overlay", 10, 100, 2000, 100);
	//
	//CapacityTests::ModelPerfomanceTest2D(
	//	[](int dimX, int dimY) -> IterativeModel* { return InitRoutines::GetInitedHeat2D(dimX, dimY); },
	//	"gpu_sweep_linedevide", 10, 100, 2000, 100);

	//CapacityTests::ModelPerfomanceTest2D(
	//	[](int dimX, int dimY) -> IterativeModel* { return InitRoutines::GetInitedHeat2D(dimX, dimY); },
	//	"gpu_sweep_blockdevide", 10, 100, 2000, 100);

	//#pragma endregion

	//#pragma region Heat3D perfomance

	//std::cout << "Heat conductivity 3D model perfomance test" << std::endl;

	//CapacityTests::ModelPerfomanceTest3D(
	//	[](int dimX, int dimY, int dimZ) -> IterativeModel* { return InitRoutines::GetInitedHeat3D(dimX, dimY, dimZ); },
	//	"cpu_gaussseidel", 10, 100, 2000, 100);

	//CapacityTests::ModelPerfomanceTest3D(
	//	[](int dimX, int dimY, int dimZ) -> IterativeModel* { return InitRoutines::GetInitedHeat3D(dimX, dimY, dimZ); },
	//	"gpu_gaussseidel_direct", 10, 100, 2000, 100);

	//CapacityTests::ModelPerfomanceTest3D(
	//	[](int dimX, int dimY, int dimZ) -> IterativeModel* { return InitRoutines::GetInitedHeat3D(dimX, dimY, dimZ); },
	//	"cpu_sweep", 10, 100, 2000, 100);

	//CapacityTests::ModelPerfomanceTest3D(
	//	[](int dimX, int dimY, int dimZ) -> IterativeModel* { return InitRoutines::GetInitedHeat3D(dimX, dimY, dimZ); },
	//	"gpu_sweep_linedevide", 10, 100, 2000, 100);

	//#pragma endregion

	//#pragma region Forest fire perfomance

	//std::cout << "Forest fire model perfomance test" << std::endl;

	//CapacityTests::ModelPerfomanceTest2D(
	//	[](int dimX, int dimY) -> IterativeModel* { return InitRoutines::GetInitedForestFire(dimX, dimY); },
	//	"cpu", 10, 100, 2000, 100);

	//CapacityTests::ModelPerfomanceTest2D(
	//	[](int dimX, int dimY) -> IterativeModel* { return InitRoutines::GetInitedForestFire(dimX, dimY); },
	//	"gpu", 10, 100, 2000, 100);

	//#pragma endregion

	//#pragma region Oil spill perfomance

	//std::cout << "Oil spill model perfomance test" << std::endl;

	//CapacityTests::ModelPerfomanceTest2D(
	//	[](int dimX, int dimY) -> IterativeModel* { return InitRoutines::GetInitedOilSpill(dimX, dimY); },
	//	"cpu", 10, 100, 2000, 100);

	//CapacityTests::ModelPerfomanceTest2D(
	//	[](int dimX, int dimY) -> IterativeModel* { return InitRoutines::GetInitedOilSpill(dimX, dimY); },
	//	"gpu", 10, 100, 2000, 100);

	//#pragma endregion

	//#pragma region Air flow perfomance

	//std::cout << "Air flow model perfomance test" << std::endl;

	//CapacityTests::ModelPerfomanceTest3D(
	//	[](int dimX, int dimY, int dimZ) -> IterativeModel* { return InitRoutines::GetInitedAirFlow(dimX, dimY, dimZ); },
	//	"cpu", 10, 100, 2000, 100);

	//CapacityTests::ModelPerfomanceTest3D(
	//	[](int dimX, int dimY, int dimZ) -> IterativeModel* { return InitRoutines::GetInitedAirFlow(dimX, dimY, dimZ); },
	//	"gpu", 10, 100, 2000, 100);

	//#pragma endregion

	//#pragma region Forest fire real/model time

	//std::cout << "Forest fire real and model time relation test" << std::endl;

	//CapacityTests::RealAndModelTimeRelation2D(
	//	[](int dimX, int dimY) -> IterativeModel* { return InitRoutines::GetInitedForestFire(dimX, dimY); },
	//	"cpu", 3600, 600, 100, 500, 100);

	//CapacityTests::RealAndModelTimeRelation2D(
	//	[](int dimX, int dimY) -> IterativeModel* { return InitRoutines::GetInitedForestFire(dimX, dimY); },
	//	"gpu", 3600, 600, 100, 500, 100);

	//#pragma endregion

	//#pragma region Oil spill real/model time

	//std::cout << "Oil spill real and model time relation test" << std::endl;

	//CapacityTests::RealAndModelTimeRelation2D(
	//	[](int dimX, int dimY) -> IterativeModel* { return InitRoutines::GetInitedOilSpill(dimX, dimY); },
	//	"cpu", 3600, 600, 100, 500, 100);

	//CapacityTests::RealAndModelTimeRelation2D(
	//	[](int dimX, int dimY) -> IterativeModel* { return InitRoutines::GetInitedOilSpill(dimX, dimY); },
	//	"gpu", 3600, 600, 100, 500, 100);

	//#pragma endregion

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
	} catch(std::exception e) {
		std::cout << e.what() << std::endl;
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
	if (CEvent == CTRL_CLOSE_EVENT)
		CloseApplication();

  return TRUE;
}

void GetButtonForManageIteration(size_t methodOrderNum, std::string& singleIterationButton, std::string& processIterationsButton) {
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
	} catch(std::exception e) {
		std::cout << "Ошибка получения прототипа модели: " << e.what() << std::endl;
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

Rect<float> GetRegionsClearence(const vector<Region>& regions) {
	Rect<float> result = regions[0].GetClearanceBorders();

	Rect<float> curClearence;
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

Rect<float> MeterClearenceByGradusArea(const Rect<float>& gradusArea) {
	Rect<float> result(Point<float>(0.0f, 0.0f), Point<float>(0.0f, 0.0f));

	float earthC = 2.0f * PI * EARTH_RADIUS;
	Point<float> oneGradInKm(
		1000.0f * (earthC * cosf(LattitudeCloserEquator(gradusArea.point1.y, gradusArea.point2.y) * (PI / 180.0f))) / 360.0f,
		1000.0f * (earthC / 360.0f));

	result.point2.x = abs(gradusArea.point1.x - gradusArea.point2.x) * oneGradInKm.x;// * 2.0f;
	result.point2.y = abs(gradusArea.point1.y - gradusArea.point2.y) * oneGradInKm.y;// * 2.0f;

	return result;
}

void DiscretizeRegions(const vector<Region>& regions, const Point<float>& luPoint,
											 float h, HostData2D<bool>& discreteField) {
	discreteField.Fill(false);
	for (auto region : regions) {
		Vectorizator::DiscretizeAndAppend(region.outerBounder, luPoint, h, true, discreteField);
		for (auto innerBounder : region.innerBounders)
			Vectorizator::DiscretizeAndAppend(innerBounder, luPoint, h, false, discreteField);
	}
}

void NormalizeRegions(vector<Region>& regions, const Rect<float>& src, const Rect<float>& dst) {
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

vector<Region> GetForestFireForecast(const ForestFireConsts& consts,
																		 vector<Region> forestRegions, vector<Region> fireRegions,
																		 float fireplaceTemperatureK, float forestRoFuel, float nonForestRoFuel,
																		 float minFireTemperatureK, float modelingTimeSec, bool isUseGpu) {
	//Получение габаритов леса градусах и метрах
	Rect<float> gradForestClearence = GetRegionsClearence(forestRegions);
	Rect<float> meterForestClearence = MeterClearenceByGradusArea(gradForestClearence);

	//Нормализация координат между загруженными из файлов и более приемлемыми для расчетов
	NormalizeRegions(forestRegions, gradForestClearence, meterForestClearence);
	NormalizeRegions(fireRegions, gradForestClearence, meterForestClearence);
	
	size_t dimX = (size_t)((meterForestClearence.point2.x - meterForestClearence.point1.x) / consts.H);
	size_t dimY = (size_t)((meterForestClearence.point2.y - meterForestClearence.point1.y) / consts.H);

	std::cout << "dimX = " << dimX << "; dimY = " << dimY << std::endl;

	ForestFireDataH data(dimX, dimY);
	HostData2D<bool> forestDiscreteGrid(dimX, dimY), fireDiscreteGrid(dimX, dimY);
	DiscretizeRegions(forestRegions, meterForestClearence.point1, consts.H, forestDiscreteGrid);
	DiscretizeRegions(fireRegions, meterForestClearence.point1, consts.H, fireDiscreteGrid);

	for (size_t x = 0; x < data.dimX(); x++) {
		for (size_t y = 0; y < data.dimY(); y++) {
			data.roFuel(x, y) = forestDiscreteGrid(x, y) ? forestRoFuel : nonForestRoFuel;
			data.t(x, y) = fireDiscreteGrid(x, y) ? fireplaceTemperatureK : consts.EnviromentTemperature;
		}
	}

	model = new ForestFireModel(consts, data);
	size_t countIteration = modelingTimeSec / consts.Tau;
	for (size_t iterNum = 0; iterNum < countIteration; iterNum++)
		model->NextIteration(isUseGpu ? "gpu" : "cpu");

	model->SynchronizeWithGpu();

	for (size_t x = 0; x < data.dimX(); x++)
		for (size_t y = 0; y < data.dimY(); y++)
			fireDiscreteGrid(x, y) = (data.t(x, y) >= minFireTemperatureK);

	auto forecastPolygons = Vectorizator::Vectorize(fireDiscreteGrid, meterForestClearence.point1, consts.H);

	vector<Region> forecastRegions;
	for (auto fireForecastPolygon : forecastPolygons)
		forecastRegions.push_back(Region(fireForecastPolygon, std::vector<DataVisualization::Geometry::Polygon<float>>(0)));

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
		std::vector<Region> forestRegions = DataVisualization::Kml::LoadPolygonsFromFile("forest.kml");
		std::vector<Region> fireRegions = DataVisualization::Kml::LoadPolygonsFromFile("fire.kml");

		ForestFireConsts consts;
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
		vector<Region> forecastRegions
			= GetForestFireForecast(consts, forestRegions, fireRegions, fireplaceTemperature, forestRoFuel,
															nonForestRoFuel, minFireTemperature, modelingTime, isUseGpu);

		std::cout << "Сохранение файла..." << std::endl;
		DataVisualization::Kml::SavePolygonsToFile(ForecastKmlFileName(), "forecastTemplate.kml", forecastRegions);

#pragma region Fire KML visualization
		Rect<float> forestClearence = GetRegionsClearence(forestRegions);
		FireKmlVisualizator visualizator(consts.H, forestClearence,
			Rect<float>(Point<float>(-30.0f, -30.0f), Point<float>(30.0f, 30.0f)));
		visualizator.fireRegions = fireRegions;
		visualizator.forestRegions = forestRegions;
		visualizator.forecastRegions = forecastRegions;

		grEngine = GraphicMgr::New(argc, argv, "Iterative models", false, false);
		grEngine->AddPresentObj(dynamic_cast<IPresentable *>(&visualizator));

		grEngine->AddUpdateKeyboardHandler('0', Action_Close);
		std::cout	<< "0 - закрыть приложение" << std::endl;

		grEngine->Run();
#pragma endregion

	} catch(DataVisualization::DataVisualizationException e) {
		std::cout << "Ошибка в модуле визуализации: " << e.what() << std::endl;
	} catch(Sonsode::SonsodeException e) {
		std::cout << "Ошибка в вычислениях: "	<< e.what() << std::endl;
	} catch(std::exception e) {
		std::cout << "Неизвестная ошибка: " << e.what() << std::endl;
	}
}
#pragma endregion

#pragma region Oil spillage
vector<Region> GetOilSpillageForecast(const OilSpillConsts& consts,
																			std::vector<Region> waterRegions, std::vector<Region> oilRegions,
																			float oilplaceImpurity, float waterDeep, float nonWaterDeep,
																			float minImpurity, float modelingTimeSec, bool isUseGpu,
																			HostData2D<bool>& waterDiscreteGrid, HostData2D<bool>& oilDiscreteGrid,
																			HostData2D<bool>& forecastDiscreteGrid,
																			Rect<float>& meterClearence) {
	//Получение габаритов водоема в градусах и метрах
	Rect<float> gradWaterClearence = GetRegionsClearence(waterRegions);
	Rect<float> meterWaterClearence = MeterClearenceByGradusArea(gradWaterClearence);

	//!!!
	meterClearence = meterWaterClearence;

	//Нормализация координат между загруженными из файлов и более приемлемыми для расчетов
	NormalizeRegions(waterRegions, gradWaterClearence, meterWaterClearence);
	NormalizeRegions(oilRegions, gradWaterClearence, meterWaterClearence);

	size_t dimX = (size_t)((meterWaterClearence.point2.x - meterWaterClearence.point1.x) / consts.H);
	size_t dimY = (size_t)((meterWaterClearence.point2.y - meterWaterClearence.point1.y) / consts.H);

	std::cout << "dimX = " << dimX << "; dimY = " << dimY << std::endl;

	OilSpillDataH data(dimX, dimY);
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

	model = new OilSpillModel(consts, data);
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

	vector<Region> forecastRegions;
	for (auto oilForecastPolygon : forecastPolygons)
		forecastRegions.push_back(Region(oilForecastPolygon, std::vector<DataVisualization::Geometry::Polygon<float>>(0)));

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
		std::vector<Region> waterRegions = DataVisualization::Kml::LoadPolygonsFromFile("water.kml");
		std::vector<Region> oilRegions = DataVisualization::Kml::LoadPolygonsFromFile("oil.kml");

		OilSpillConsts consts;
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
		Rect<float> meterClearence;

		vector<Region> forecastRegions
			= GetOilSpillageForecast(consts, waterRegions, oilRegions, oilplaceImpurity, waterDeep,
															 nonWaterDeep, minOilImpurity, modelingTime, isUseGpu,
															 isWaterField, isOilField, isForecastField, meterClearence); //!!!

		std::cout << "Сохранение файла..." << std::endl;
		DataVisualization::Kml::SavePolygonsToFile(ForecastKmlFileName(), "forecastTemplate.kml", forecastRegions);

#pragma region Oil KML visualization
		DataVisualization::Geometry::Rect<float> waterClearence = GetRegionsClearence(waterRegions);

		Point<float> h(
			fabsf(waterClearence.point1.x - waterClearence.point2.x) / (float)isWaterField.dimX(),
			fabsf(waterClearence.point1.y - waterClearence.point2.y) / (float)isWaterField.dimY());

		OilKmlVisualizator visualizator(h, waterClearence,
			Rect<float>(Point<float>(-30.0f, -30.0f), Point<float>(30.0f, 30.0f)));
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

	} catch(DataVisualization::DataVisualizationException e) {
		std::cout << "Ошибка в модуле визуализации: " << e.what() << std::endl;
	} catch(Sonsode::SonsodeException e) {
		std::cout << "Ошибка в вычислениях: "	<< e.what() << std::endl;
	} catch(std::exception e) {
		std::cout << "Неизвестная ошибка: " << e.what() << std::endl;
	}
}
#pragma endregion

#pragma endregion