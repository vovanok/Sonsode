#include "CapacityTests.h"

namespace CapacityTests {
	namespace {
		void TimeByFloat(float value, size_t& hour, size_t& min, size_t& sec) {
			hour = (size_t)(value / 3600.0f);
			min = (size_t)((value - (hour * 3600.0f)) / 60.0f);
			sec = (size_t)(value - (hour * 3600.0f) - (min * 60.0f));
		}
	}

	void HeatConductivity2D(size_t countIteration, size_t startDim, size_t finishDim, size_t stepDim, std::string methodName) {
		double iterationTime;
		double sumTime;
		size_t iterationNumber;
	
		std::cout << "Тест производительности 2D модели теплопроводности. Метод: " << methodName << std::endl;

		HeatConductivity2DModelTest *model;
		
		for (size_t dim = startDim; dim <= finishDim; dim += stepDim) {
			model = InitRoutines::GetInitedHeatConductivityModel(dim, dim);
		
			try {
				sumTime = 0;
		
				for (size_t iterNum = 0; iterNum < countIteration + 2; iterNum++) {
					model->NextIteration(methodName, iterationTime, iterationNumber);
					if (iterNum != 0 && iterNum != countIteration + 1)
						sumTime += iterationTime;
				}

				std::cout << "Размерность задачи: " << dim << "x" << dim 
					<< ".Среднее время на итерацию: " << sumTime / countIteration << "\n";
			} catch(std::string e) {
				std::cout << e << std::endl;
			}

			delete model;
			model = 0;
		}

		std::cout << "Расчет закончен\n";
	}

	void HeatConductivity3D(size_t countIteration, size_t startDim, size_t finishDim, size_t stepDim, std::string methodName) {
		double iterationTime;
		double sumTime;
		size_t iterationNumber;
	
		std::cout << "Тест производительности 3D модели теплопроводности. Метод: " << methodName << std::endl;

		HeatConductivity3DModelTest *model;

		for (size_t dim = startDim; dim <= finishDim; dim += stepDim) {
			model = InitRoutines::GetInitedHeatConductivity3DModel(dim, dim, dim);
		
			try {
				sumTime = 0;
		
				for (size_t iterNum = 0; iterNum < countIteration + 2; iterNum++) {
					model->NextIteration(methodName, iterationTime, iterationNumber);
					if (iterNum != 0 && iterNum != countIteration + 1)
						sumTime += iterationTime;
				}

				std::cout << "Размерность задачи: " << dim << "x" << dim << "x" << dim
					<< ".Среднее время на итерацию: " << sumTime / countIteration << "\n";
			} catch(std::string e) {
				std::cout << e << std::endl;
			}

			delete model;
			model = 0;
		}

		std::cout << "Расчет закончен" << std::endl;
	}

	void OilSpillage(size_t countIteration, size_t startDim, size_t finishDim, size_t stepDim, std::string methodName) {
		double iterationTime;
		double sumTime;
		size_t iterationNumber;
	
		std::cout << "Тест производительности 2D модели нефтеразлива. Метод: " << methodName << std::endl;

		OilSpillageImprovedModelTest *model;
		
		for (size_t dim = startDim; dim <= finishDim; dim += stepDim) {
			model = InitRoutines::GetInitedOilSpillageImprovedModel(dim, dim);
		
			try {
				sumTime = 0;
		
				for (size_t iterNum = 0; iterNum < countIteration + 2; iterNum++) {
					model->NextIteration(methodName, iterationTime, iterationNumber);
					if (iterNum != 0 && iterNum != countIteration + 1)
						sumTime += iterationTime;
				}

				std::cout << "Размерность задачи: " << dim << "x" << dim 
					<< ".Среднее время на итерацию: " << sumTime / countIteration << "\n";
			} catch(std::string e) {
				std::cout << e << std::endl;
			}

			delete model;
			model = 0;
		}

		std::cout << "Расчет закончен\n";
	}
	
	void AirFlow(size_t countIteration, size_t startDim, size_t finishDim, size_t stepDim, std::string methodName) {
		double iterationTime;
		double sumTime;
		size_t iterationNumber;
	
		std::cout << "Тест производительности 3D модели воздушных потоков. Метод: " << methodName << std::endl;

		AirFlowDistributionModelTest* model;
		
		for (size_t dim = startDim; dim <= finishDim; dim += stepDim) {
			model = InitRoutines::GetInitedAirFlowDistributionModel(dim, dim, dim);
		
			try {
				sumTime = 0;
		
				for (size_t iterNum = 0; iterNum < countIteration + 2; iterNum++) {
					model->NextIteration(methodName, iterationTime, iterationNumber);
					if (iterNum != 0 && iterNum != countIteration + 1)
						sumTime += iterationTime;
				}

				std::cout << "Размерность задачи: " << dim << "x" << dim << "x" << dim 
					<< ".Среднее время на итерацию: " << sumTime / countIteration << "\n";
			} catch(std::string e) {
				std::cout << e << std::endl;
			}

			delete model;
			model = 0;
		}

		std::cout << "Расчет закончен\n";
	}

	void FireSpread() {
		size_t countIteration = 20;
		double iterationTime;
		double sumTimeCPU;
		double sumTimeGPU;
		size_t iterationNumber;
	
		FireSpreadSimpleModelTest *model;

		for (int dim = 100; dim <= 2000; dim += 100) {
			model = InitRoutines::GetInitedFireSpreadSimpleModel(dim, dim);

			model->GpuOff();
			sumTimeCPU = 0.0f;
			for (size_t iterNum = 0; iterNum < countIteration; iterNum++) {
				model->NextIteration("cpu_gaussseidel", iterationTime, iterationNumber);
				sumTimeCPU += iterationTime;
			}

			model->GpuOn();
			sumTimeGPU = 0.0f;
			for (size_t iterNum = 0; iterNum < countIteration; iterNum++) {
				model->NextIteration("cpu_gaussseidel", iterationTime, iterationNumber);
				sumTimeGPU += iterationTime;
			}

			std::cout << "Размерность задачи: " << dim << "x" << dim 
				<< ". CPU: " << sumTimeCPU / countIteration
				<< "; GPU: " << sumTimeGPU / countIteration << std::endl;

			//model->DeInitGpu();

			delete model;
			model = NULL;
		}

		std::cout << "Расчет закончен\n";
	}


	void FireSpreadRealAndModelTimeRelation(size_t maxRealTimeSec, size_t stepRealTimeSec,
			size_t startDim, size_t finishDim, size_t stepDim, std::string methodName) {
		
		double iterationTime;
		size_t iterationNumber;
	
		std::cout << "Тест производительности модели лесного пожара. Метод: " << methodName << std::endl;

		FireSpreadSimpleModelTest *model;
		
		for (size_t dim = startDim; dim <= finishDim; dim += stepDim) {
			model = InitRoutines::GetInitedFireSpreadSimpleModel(dim, dim);
			std::cout << dim << "x" << dim;

			float calculationTime = 0.0f;
			float nextModelingTimeFrontier = stepRealTimeSec;

			while (model->currentTime() <= maxRealTimeSec) {
				try {
					model->NextIteration(methodName, iterationTime, iterationNumber);
					calculationTime += iterationTime;

					if (model->currentTime() >= nextModelingTimeFrontier) {
						std::cout << " " << std::fixed << std::setprecision(4) << calculationTime;
						nextModelingTimeFrontier += stepRealTimeSec;
					}
				} catch(std::string e) {
					std::cout << std::endl << e << std::endl;
				}
			}

			std::cout << std::endl;

			delete model;
			model = 0;
		}

		std::cout << "Расчет закончен\n";
	}

	void OilSpillageRealAndModelTimeRelation(size_t maxModelingTimeSec, size_t modelingTimeStepSec,
			size_t startDim, size_t finishDim, size_t stepDim, std::string methodName) {
		double iterationTime;
		size_t iterationNumber;
	
		std::cout << "Тест производительности модели разлива нефти. Метод: " << methodName << std::endl;

		OilSpillModel* model;
		
		for (size_t dim = startDim; dim <= finishDim; dim += stepDim) {
			model = InitRoutines::GetInitedOilSpillageImprovedModel(dim, dim);
			std::cout << dim << "x" << dim;

			float calculationTime = 0.0f;
			float nextModelingTimeFrontier = modelingTimeStepSec;

			while (model->currentTime() <= maxModelingTimeSec) {
				try {
					model->NextIteration(methodName, iterationTime, iterationNumber);
					calculationTime += iterationTime;

					if (model->currentTime() >= nextModelingTimeFrontier) {
						std::cout << " " << std::fixed << std::setprecision(4) << calculationTime;
						nextModelingTimeFrontier += modelingTimeStepSec;
					}
				} catch(std::string e) {
					std::cout << std::endl << e << std::endl;
				}
			}

			std::cout << std::endl;

			delete model;
			model = 0;
		}

		std::cout << "Расчет закончен\n";
	}
}