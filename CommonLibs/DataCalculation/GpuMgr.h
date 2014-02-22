#pragma once

#include <cuda_runtime.h>
#include <iostream>

enum MemCpyDir
{
	DtH = 0,
	HtD = 1,
	HtH = 2,
	DtD = 3
};

class GpuMgr
{
private:
	static GpuMgr *Instance;
	GpuMgr();
	~GpuMgr();
public:
	static GpuMgr *GetInstance();
	static void FreeInstance();

	int *DevicesNums;
	bool *DevicesIsSet;
	int CountDevices;
	
	void SetDevice(int);
	void Malloc(void**, int);
	void MemCpy(void*, void*, int, MemCpyDir);
	void Free(void *);
	void ThreadSynchronize();
	void CheckLastError();
};