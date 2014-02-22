#pragma once

namespace Sonsode {
	template<class T> class SweepFactors {
	public:
		T L, M;

		__host__ __device__ SweepFactors() : L(0), M(0) { }
		__host__ __device__ SweepFactors(const T& l, const T& m) : L(l), M(m) { }
		__host__ __device__ SweepFactors<T> GetNextFactors(const T& q, const T& alpha, const T& beta, const T& gamma);

	private:
		__host__ __device__ T CalculateL(const T& prevL, const T& alpha, const T& beta, const T& gamma);
		__host__ __device__ T CalculateM(const T& prevL, const T& prevM, const T& q, const T& alpha, const T& beta, const T& gamma);
	};

	#pragma region Implementation

	template<class T> SweepFactors<T> SweepFactors<T>::GetNextFactors(const T& q, const T& alpha, const T& beta, const T& gamma) {
		return SweepFactors<T>(CalculateL(L, alpha, beta, gamma), CalculateM(L, M, q, alpha, beta, gamma));
	}

	template<class T> T SweepFactors<T>::CalculateL(const T& prevL, const T& alpha, const T& beta, const T& gamma) {
		return -gamma / (beta + alpha * prevL);
	}

	template<class T> T SweepFactors<T>::CalculateM(const T& prevL, const T& prevM, const T& q, const T& alpha, const T& beta, const T& gamma) {
		return (q - alpha * prevM) / (beta + alpha * prevL);
	}

	#pragma endregion
};