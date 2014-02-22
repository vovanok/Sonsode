#include "GpuMgr.h"

//Self-sufficiency
GpuMgr *GpuMgr::Instance = 0;
GpuMgr *GpuMgr::GetInstance()
{
	if (GpuMgr::Instance == 0)
		GpuMgr::Instance = new GpuMgr();
	return GpuMgr::Instance;
}

void GpuMgr::FreeInstance()
{
	if (GpuMgr::Instance != 0)
		delete GpuMgr::Instance;
}

//constructor & destructor
GpuMgr::GpuMgr()
{
	int count = 0;
	int num = -1;

	this->CountDevices = 0;
	
	cudaGetDeviceCount(&count);
	if(count == 0)
	{
		std::cout << "Нет устройств, поддерживающих CUDA\n";
		return;
	}

//	this->DevicesNums = new int[count];

	for(int i = 0; i < count; i++)
	{
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess)
		{
			if (prop.major >= 1)
			{
				//this->DevicesNums[this->CountDevices] = i;
				float *tmp = 0;
				cudaSetDevice(i);
				//StopProfile();
				Malloc((void**)&tmp, 1);
				Free(tmp);
				this->CountDevices++;
			}
		}
		else
		{
			std::cout << "Ошибка получения свойств устройства с номером " << i << "\n";
		}
	}

	if(this->CountDevices == 0)
	{
		std::cout << "Нет устройств с CC >= 1.0" << "\n";
		return;
	}

	//this->DevicesIsSet = new bool[this->CountDevices];

	//logMsg << "Номера устройств, поддерживащих CUDA: ";
	//for (int i = 0; i < this->CountDevices; i++)
	//{
	//	this->DevicesIsSet[i] = false;
	//	logMsg << this->DevicesNums[i] << " ";
	//}
	//logMsg << "\n";
}

GpuMgr::~GpuMgr()
{
	cudaDeviceReset();
	//cudaThreadExit();//cutilExit(0, new char*[0]);
	std::cout << "CUDA деинициализирована.\n";
}

//API
void GpuMgr::SetDevice(int ccDeviceNum)
{
	if (ccDeviceNum >= this->CountDevices)
	{
		std::cout << "Устройство с номером " << ccDeviceNum << " не существует." << "\n";
		return;
	}

	cudaError_t error = cudaSetDevice(ccDeviceNum);

	if (error != cudaSuccess)
		std::cout << "Ошибка инициализации устройства с номером " << ccDeviceNum << ": " << cudaGetErrorString(error) << "\n";

	//if (!this->DevicesIsSet[ccDeviceNum])
	//{
		//cudaError_t error = cudaSetDevice(this->DevicesNums[ccDeviceNum]);
		//if (error != CUDA_SUCCESS)
		//{
		//	logMsg << "Устройство с номером " << ccDeviceNum << "не инициализировано." << "\n";
		//}
		//else
		//{
		//	this->DevicesIsSet[ccDeviceNum] = true;
		//	logMsg << "Устройство с номером " << ccDeviceNum << " инициализировано" << "\n";
		//}
	//}
}

void GpuMgr::Malloc(void **data, int countBytes)
{
	cudaError_t error = cudaMalloc(data, countBytes);

	if (error != cudaSuccess)
		std::cout << "Ошибка выделения памяти CUDA: " << cudaGetErrorString(error) << "\n";
}

void GpuMgr::Free(void *data)
{
	cudaError_t error = cudaFree(data);

	if (error != cudaSuccess)
		std::cout << "Ошибка очистки памяти CUDA: " << cudaGetErrorString(error) << "\n";
}

void GpuMgr::MemCpy(void *destination, void *source, int countBytes, MemCpyDir direction)
{
	cudaMemcpyKind memcpyKind = cudaMemcpyHostToHost;

	switch (direction)
	{
		case DtH:
			memcpyKind = cudaMemcpyDeviceToHost;
			break;
		case HtD:
			memcpyKind = cudaMemcpyHostToDevice;
			break;
		case HtH:
			memcpyKind = cudaMemcpyHostToHost;
			break;
		case DtD:
			memcpyKind = cudaMemcpyDeviceToDevice;
			break;
	}

	cudaError_t error = cudaMemcpy(destination, source, countBytes, memcpyKind);
	if (error != cudaSuccess)
		std::cout << "Ошибка копирования памяти: " << cudaGetErrorString(error) 
			<< " Количество: " << countBytes 
			<< ". Направление: " << direction << "." << "\n";
}

void GpuMgr::ThreadSynchronize()
{
	cudaError_t error = cudaThreadSynchronize();

	if (error != cudaSuccess)
		std::cout << "Ошибка синхронизации с ядром: " << cudaGetErrorString(error) << "\n";
}

void GpuMgr::CheckLastError()
{
	//##RETURN VALUE
	cudaError_t lastError = cudaGetLastError();
	if (lastError != cudaSuccess)
		std::cout << "Последняя ошибка: " << cudaGetErrorString(lastError) << "\n";
}