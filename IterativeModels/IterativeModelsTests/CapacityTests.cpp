#include "CapacityTests.h"

namespace CapacityTests {
	void ModelPerfomanceTest2D(std::function<IterativeModel*(size_t, size_t)> modelGetter, std::string methodName,
			size_t countIteration, size_t startDim, size_t finishDim, size_t stepDim) {
		std::cout << "Method " << methodName << std::endl;

		double iterationTime;
		double sumTime;
		size_t iterationNumber;
	
		IterativeModel* model;

		for (size_t dim = startDim; dim <= finishDim; dim += stepDim) {
			model = modelGetter(dim, dim);
		
			try {
				sumTime = 0;
		
				for (size_t iterNum = 0; iterNum < countIteration + 2; iterNum++) {
					model->NextIteration(methodName, iterationTime, iterationNumber);
					if (iterNum != 0 && iterNum != countIteration + 1)
						sumTime += iterationTime;
				}

				std::cout << dim << "x" << dim << ": Time = " << sumTime / countIteration << "\n";
			} catch(std::exception e) {
				std::cout << e.what() << std::endl;
			}

			delete model;
			model = 0;
		}

		std::cout << "Calculates end" << std::endl;
	}

	void ModelPerfomanceTest3D(std::function<IterativeModel*(size_t, size_t, size_t)> modelGetter, std::string methodName,
			size_t countIteration, size_t startDim, size_t finishDim, size_t stepDim) {
		std::cout << "Method " << methodName << std::endl;

		double iterationTime;
		double sumTime;
		size_t iterationNumber;
	
		IterativeModel* model;

		for (size_t dim = startDim; dim <= finishDim; dim += stepDim) {
			model = modelGetter(dim, dim, dim);
		
			try {
				sumTime = 0;
		
				for (size_t iterNum = 0; iterNum < countIteration + 2; iterNum++) {
					model->NextIteration(methodName, iterationTime, iterationNumber);
					if (iterNum != 0 && iterNum != countIteration + 1)
						sumTime += iterationTime;
				}

				std::cout << dim << "x" << dim << "x" << dim << ": Time = " << sumTime / countIteration << "\n";
			} catch(std::exception e) {
				std::cout << e.what() << std::endl;
			}

			delete model;
			model = 0;
		}

		std::cout << "Calculates end" << std::endl;
	}

	void RealAndModelTimeRelation2D(std::function<IterativeModel*(size_t, size_t)> modelGetter, std::string methodName,
			size_t maxTimeSec, size_t stepTimeSec, size_t startDim, size_t finishDim, size_t stepDim) {
		std::cout << "Method " << methodName << std::endl;
				
		double iterationTime;
		size_t iterationNumber;
		IterativeModel* model;

		for (size_t dim = startDim; dim <= finishDim; dim += stepDim) {
			model = modelGetter(dim, dim);
			std::cout << dim << "x" << dim;

			float calculationTime = 0.0f;
			float nextModelingTimeFrontier = stepTimeSec;

			while (model->currentTime() <= maxTimeSec) {
				try {
					model->NextIteration(methodName, iterationTime, iterationNumber);
					calculationTime += iterationTime;

					if (model->currentTime() >= nextModelingTimeFrontier) {
						std::cout << " " << std::fixed << std::setprecision(4) << calculationTime;
						nextModelingTimeFrontier += stepTimeSec;
					}
				} catch(std::exception e) {
					std::cout << std::endl << e.what() << std::endl;
				}
			}

			std::cout << std::endl;

			delete model;
			model = 0;
		}

		std::cout << "Calculates end" << std::endl;
	}
}