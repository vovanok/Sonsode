#include "FireSpreadSimpleModel.h"

#pragma region Device routines
//Функция вычисления противоточной производной
__device__ static float Get_t4(float tem_self, float tem_right, float tem_left,
		float tem_up, float tem_down, Vector2D<float> u, float r0, float danu, float hh) {
	return (danu / hh) * (tem_right + tem_left + tem_up + tem_down - 4 * tem_self)
		- u.x * (u.x > 0 ? r0 * (tem_self - tem_left) : r0 * (tem_right - tem_self))
		- u.y * (u.y > 0 ? r0 * (tem_self - tem_down) : r0 * (tem_up - tem_self));
}

__device__ static float Get_q4(float tem_self, float tem_prev, float tem_next, float t4_self, float ap, float tau) {
	return - ap * tem_prev + (1 + 2 * ap) * tem_self - ap * tem_next + tau * t4_self;
}

//Формула для вычисления коэффициентов при прямой прогонке
__device__ static float Get_m4(float m4_prev, float l_self, float q4_prev, float ap) {
	return (q4_prev / ap + m4_prev) * l_self;
}

//Формула для вычисления температуры при обратной прогонке
__device__ static float Get_tem(float l_next, float tem_next, float m4_next) {
	return l_next * tem_next + m4_next;
}

__device__ static float Get_fgor(float tem_self, float temKr, float qbig, int mstep, float tzv) {
	return tem_self < temKr ? 0 : (qbig * pow(fabs(tem_self), mstep) * exp(- tzv / tem_self));
}

//Получение новой плотности топлива при горении
//__device__ static float Get_rotop(float rotop_old, float tau, float fgor) {
//	return rotop_old / (1 + tau * fgor);
//}

//Получение температуры после горения
__device__ static float Get_fire_tem(float tem_self, float rotop_self,
																		 float tau, float qlitl, float fgor, float ks, float vlag) {
	return tem_self + tau * qlitl * rotop_self * fgor / (1 + ks * vlag);
}
#pragma endregion

#pragma region Kernels
//Ядро вычиления противоточных производных
__global__ static void Kernel_FireSpreadSimpleModel_CounterflowDerivative(FireSpreadDataD data_dev,
		Vector2D<float> u, float r0, float danu, float hh) {

	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ float subTem [BLOCK_SIZE_FSSM + 2][BLOCK_SIZE_FSSM + 2];
	//перемещаю свою точку в разделяемую память
	subTem[threadIdx.x + 1][threadIdx.y + 1] =
		(x <= data_dev.dimX() - 1 && y <= data_dev.dimY() - 1) ? data_dev.t(x, y) : 0.0f;

	//если я на границе, перемещаю граничные точки в разделяемую память
	if (blockIdx.x != 0 && threadIdx.x == 0) //transport data for left border
		subTem[threadIdx.x][threadIdx.y + 1] = data_dev.t(x - 1, y);
	if (blockIdx.y != 0 && threadIdx.y == 0) //transport data for down border
		subTem[threadIdx.x + 1][threadIdx.y] = data_dev.t(x, y - 1);
	if (blockIdx.x != gridDim.x - 1 && threadIdx.x == blockDim.x - 1) //transport data for right border
		subTem[threadIdx.x + 2][threadIdx.y + 1] = data_dev.t(x + 1, y);
	if (blockIdx.y != gridDim.y - 1 && threadIdx.y == blockDim.y - 1) //transport data for up border
		subTem[threadIdx.x + 1][threadIdx.y + 2] = data_dev.t(x, y + 1);
	__syncthreads();

	//если моя точка внутренняя и "белая" на шахматной доске
	if ((x != 0 && x != data_dev.dimX() - 1 && y != 0 && y != data_dev.dimY() - 1)
			&& ((threadIdx.x + threadIdx.y) % 2 == 0)) {
		//вычисляю свою точку по формуле
		data_dev.t4(x, y) = 
			Get_t4(
				subTem[threadIdx.x + 1][threadIdx.y + 1],
				subTem[threadIdx.x + 2][threadIdx.y + 1],
				subTem[threadIdx.x][threadIdx.y + 1],
				subTem[threadIdx.x + 1][threadIdx.y + 2],
				subTem[threadIdx.x + 1][threadIdx.y],
				u, r0, danu, hh);
	}
	__syncthreads();

	//если моя точка внутренняя и "черная" на шахматной доске
	if ((x != 0 && x != data_dev.dimX() - 1 && y != 0 && y != data_dev.dimY() - 1)
			&& ((threadIdx.x + threadIdx.y) % 2 == 1)) {
		//вычисляю свою точку по формуле
		data_dev.t4(x, y) = 
			Get_t4(
				subTem[threadIdx.x + 1][threadIdx.y + 1],
				subTem[threadIdx.x + 2][threadIdx.y + 1],
				subTem[threadIdx.x][threadIdx.y + 1],
				subTem[threadIdx.x + 1][threadIdx.y + 2],
				subTem[threadIdx.x + 1][threadIdx.y],
				u, r0, danu, hh);
	}
}

//Ядро прогонки вдоль оси X
__global__ static void Kernel_FireSpreadSimpleModel_RunAroundX(FireSpreadDataD data_dev,
		DeviceData1D<float> ly, float ap, float tau) {

	size_t y = blockIdx.x * blockDim.x + threadIdx.x;

	if (y > 0 && y < data_dev.dimY() - 1) {
		float q4_self;
		for (size_t x = 1; x < data_dev.dimX() - 1; x++) {
			q4_self = Get_q4(data_dev.t(x, y), data_dev.t(x - 1, y), data_dev.t(x + 1, y), data_dev.t4(x, y), ap, tau);
			data_dev.m(x + 1, y) = Get_m4(data_dev.m(x, y), ly(x + 1), q4_self, ap); //!!! не mx4, а m4 проверить алгоритм в разных прогонках нужно использовать разные m
		}

		for (size_t x = data_dev.dimX() - 2; x > 1; x--)
			data_dev.t(x, y) = Get_tem(ly(x + 1), data_dev.t(x + 1, y), data_dev.m(x + 1, y));
	}
}

//Ядро прогонки вдоль оси Y
__global__ static void Kernel_FireSpreadSimpleModel_RunAroundY(FireSpreadDataD data_dev,
		DeviceData1D<float> lx, float ap, float tau) {

	size_t x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x > 0 && x < data_dev.dimX() - 1) {
		float q4_self;
		for (size_t y = 1; y < data_dev.dimX() - 1; y++) 	{
			q4_self = Get_q4(data_dev.t(x, y),  data_dev.t(x, y - 1), data_dev.t(x, y + 1), data_dev.t4(x, y), ap, tau);
			data_dev.m(x, y + 1) = Get_m4(data_dev.m(x, y), lx(y + 1), q4_self, ap);
		}

		for (size_t y = data_dev.dimY() - 2; y > 1; y--)
			data_dev.t(x, y) = Get_tem(lx(y + 1), data_dev.t(x, y + 1), data_dev.m(x, y + 1));
	}
}

//Ядро вычисления температур после горения
__global__ static void Kernel_FireSpreadSimpleModel_Fire(FireSpreadDataD data_dev,
		float temKr, float qbig, int mstep, float tzv, float tau, float qlitl, float ks, float vlag) {

	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ float subTem [BLOCK_SIZE_FSSM + 2][BLOCK_SIZE_FSSM + 2];

	//перемещаю свою точку в разделяемую память
	subTem[threadIdx.x + 1][threadIdx.y + 1] =
		(x <= data_dev.dimX() - 1 && y <= data_dev.dimY() - 1) ? data_dev.t(x, y) : 0.0f;

	//если я на границе, перемещаю граничные точки в разделяемую память
	if (blockIdx.x != 0 && threadIdx.x == 0) //transport data for left border
		subTem[threadIdx.x][threadIdx.y + 1] = data_dev.t(x - 1, y);
	if (blockIdx.y != 0 && threadIdx.y == 0) //transport data for down border
		subTem[threadIdx.x + 1][threadIdx.y] = data_dev.t(x, y - 1);
	if (blockIdx.x != gridDim.x - 1 && threadIdx.x == blockDim.x - 1) //transport data for right border
		subTem[threadIdx.x + 2][threadIdx.y + 1] = data_dev.t(x + 1, y);
	if (blockIdx.y != gridDim.y - 1 && threadIdx.y == blockDim.y - 1) //transport data for up border
		subTem[threadIdx.x + 1][threadIdx.y + 2] = data_dev.t(x, y + 1);
	__syncthreads();

	//если моя точка внутренняя и "белая" на шахматной доске
	if ((x != 0 && x != data_dev.dimX() - 1 && y != 0 && y != data_dev.dimY() - 1) && ((threadIdx.x + threadIdx.y) % 2 == 0)) 	{
		float fgor = Get_fgor(subTem[threadIdx.x + 1][threadIdx.y + 1], temKr, qbig, mstep, tzv);
		data_dev.roFuel(x, y) = data_dev.roFuel(x, y) / (1.0f + tau * fgor);
		data_dev.t(x, y) = Get_fire_tem(subTem[threadIdx.x + 1][threadIdx.y + 1], data_dev.roFuel(x, y), tau, qlitl, fgor, ks, vlag);
	}
	__syncthreads();

	//если моя точка внутренняя и "черная" на шахматной доске
	if ((x != 0 && x != data_dev.dimX() - 1 && y != 0 && y != data_dev.dimY() - 1) && ((threadIdx.x + threadIdx.y) % 2 == 1)) 	{
		float fgor = Get_fgor(subTem[threadIdx.x + 1][threadIdx.y + 1], temKr, qbig, mstep, tzv);
		data_dev.roFuel(x, y) = data_dev.roFuel(x, y) / (1.0f + tau * fgor);
		data_dev.t(x, y) = Get_fire_tem(subTem[threadIdx.x + 1][threadIdx.y + 1], data_dev.roFuel(x, y), tau, qlitl, fgor, ks, vlag);
	}
}
#pragma endregion

#pragma region Kernel runs
void FireSpreadSimpleModel::Run_Kernel_FireSpreadSimpleModel_CounterflowDerivative(FireSpreadDataD data_dev,
		Vector2D<float> u, float r0, float danu, float hh) {

	size_t gridSizeX = (data_dev.dimX() / BLOCK_SIZE_FSSM) + ((data_dev.dimX() % BLOCK_SIZE_FSSM) > 0 ? 1 : 0);
	size_t gridSizeY = (data_dev.dimY() / BLOCK_SIZE_FSSM) + ((data_dev.dimY() % BLOCK_SIZE_FSSM) > 0 ? 1 : 0);

	dim3 threads (BLOCK_SIZE_FSSM, BLOCK_SIZE_FSSM);
	dim3 blocks (gridSizeX, gridSizeY);

	Kernel_FireSpreadSimpleModel_CounterflowDerivative<<<blocks, threads>>>(data_dev, u, r0, danu, hh);
}

void FireSpreadSimpleModel::Run_Kernel_FireSpreadSimpleModel_RunAroundX(FireSpreadDataD data_dev,	
		DeviceData1D<float> ly, float ap, float tau) {

	size_t gridSizeX = (data_dev.dimY() / BLOCK_SIZE_FSSM) + ((data_dev.dimY() % BLOCK_SIZE_FSSM) > 0 ? 1 : 0);

	dim3 threads (BLOCK_SIZE_FSSM);
	dim3 blocks (gridSizeX);

	Kernel_FireSpreadSimpleModel_RunAroundX<<<blocks, threads>>>(data_dev, ly, ap, tau);
}

void FireSpreadSimpleModel::Run_Kernel_FireSpreadSimpleModel_RunAroundY(FireSpreadDataD data_dev,
		DeviceData1D<float> lx, float ap, float tau) {

	size_t gridSizeX = (data_dev.dimX() / BLOCK_SIZE_FSSM) + ((data_dev.dimX() % BLOCK_SIZE_FSSM) > 0 ? 1 : 0);

	dim3 threads (BLOCK_SIZE_FSSM);
	dim3 blocks (gridSizeX);

	Kernel_FireSpreadSimpleModel_RunAroundY<<<blocks, threads>>>(data_dev, lx, ap, tau);
}

void FireSpreadSimpleModel::Run_Kernel_FireSpreadSimpleModel_Fire(FireSpreadDataD data_dev,
		float temKr, float qbig, int mstep, float tzv, float tau, float qlitl, float ks, float vlag) {

	size_t gridSizeX = (data_dev.dimX() / BLOCK_SIZE_FSSM) + ((data_dev.dimX() % BLOCK_SIZE_FSSM) > 0 ? 1 : 0);
	size_t gridSizeY = (data_dev.dimY() / BLOCK_SIZE_FSSM) + ((data_dev.dimY() % BLOCK_SIZE_FSSM) > 0 ? 1 : 0);

	dim3 threads (BLOCK_SIZE_FSSM, BLOCK_SIZE_FSSM);
	dim3 blocks (gridSizeX, gridSizeY);

	Kernel_FireSpreadSimpleModel_Fire<<<blocks, threads>>>(data_dev, temKr, qbig, mstep, tzv, tau, qlitl, ks, vlag);
}
#pragma endregion