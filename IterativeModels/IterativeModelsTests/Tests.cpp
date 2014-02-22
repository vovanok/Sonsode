#include "Tests.h"

void Tests::GpuDevice() {
	std::cout << "Start" << std::endl;

	try {
		std::cout << "Trying get all devices" << std::endl;
		auto gpus = Sonsode::GpuDeviceFactory::GpuDevices();
	
		std::cout << "Count of devices: " << gpus.size() << std::endl;
		std::cout << "Devices list:" << std::endl;
		for (auto gpu : gpus)
			std::cout << "ID: " << gpu->id() << std::endl;
		std::cout << "End of device list" << std::endl;

		if (gpus.size() > 0) {
			float *data_h = new float[10];
			float *data_d = gpus[0]->Malloc<float>(10);

			std::cout << "Source data: ";
			for (int i = 0; i < 10; i++) {
				data_h[i] = (float)(i * i);
				std::cout << data_h[i] << " ";
			}
			std::cout << std::endl;

			gpus[0]->CpyTo(data_d, data_h, 10);

			std::cout << "Zero data: ";
			for (int i = 0; i < 10; i++) {
				data_h[i] = 0;
				std::cout << data_h[i] << " ";
			}
			std::cout << std::endl;

			gpus[0]->CpyFrom(data_h, data_d, 10);

			std::cout << "Received data: ";
			for (int i = 0; i < 10; i++)
				std::cout << data_h[i] << " ";
			std::cout << std::endl;
		}
	} catch(std::string e) {
		std::cout << "Error: " << e << std::endl;
	}

	std::cout << "Finish" << std::endl;
	getchar();
}

void Tests::DeviceData() {
	try {
		size_t dimX = 10;
		size_t dimY = 10;
		Sonsode::HostData2D<float> data_h(dimX, dimY);

		std::cout << "Source host data:" << std::endl;
		for(size_t x = 0; x < dimX; x++) {
			for(size_t y = 0; y < dimY; y++) {
				data_h(x, y) = (float)(x * y + 1);
				std::cout << data_h(x, y) << " ";
			}
			std::cout << std::endl;
		}

		std::cout << "Create device data" << std::endl;
		auto gpus = Sonsode::GpuDeviceFactory::GpuDevices();
		if (gpus.size() == 0)
			throw "No GPUs";

		Sonsode::DeviceData2D<float> data_d(*gpus[0], dimX, dimY);

		std::cout << "Copy data to device" << std::endl;
		data_d.TakeFrom(data_h);

		std::cout << "Zero host data:" << std::endl;
		for(size_t x = 0; x < dimX; x++) {
			for(size_t y = 0; y < dimY; y++) {
				data_h(x, y) = 0;
				std::cout << data_h(x, y) << " ";
			}
			std::cout << std::endl;
		}

		std::cout << "Increment device data" << std::endl;
		/*gpus[0]*/data_d.gpu().SetAsCurrent();
		CuTests::RunTestKernel1(data_d);
		//CuTests::RunTestKernel2(data_d.GetData(), data_d.DimX(), data_d.DimY());
		data_d.gpu().Synchronize();
		data_d.gpu().CheckLastErr();

		std::cout << "Copy data from device" << std::endl;
		data_d.PutTo(data_h);

		data_d.Erase();
		data_d.gpu().Close();

		std::cout << "Result host data: " << std::endl;
		for(size_t x = 0; x < dimX; x++) {
			for(size_t y = 0; y < dimY; y++) {
				std::cout << data_h(x, y) << " ";
			}
			std::cout << std::endl;
		}
	} catch(std::string e) {
		std::cout << "Error: " << e << std::endl;
	}
	std::cout << "Test complete" << std::endl;
	getchar();
}

void GpuDeviceFunc(Sonsode::GpuDevice inGpuDevice) {
	std::cout << "Inner GpuDevice ID: " << inGpuDevice.id() << std::endl;
}

void Tests::GpuDevice2() {
	try {
		auto outerGpuDevice = Sonsode::GpuDeviceFactory::GetById(0);
		std::cout << "Outer GpuDevice ID: " << outerGpuDevice.id() << std::endl;
		GpuDeviceFunc(outerGpuDevice);
		std::cout << "Test complete" << std::endl;
		getchar();
	} catch (std::string e) {
		std::cout << "Test error: " << e << std::endl;
	}
}