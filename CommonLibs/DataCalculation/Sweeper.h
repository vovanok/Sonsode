#pragma once

#include "HostData.hpp"

class SweepSolveable {
protected:
	struct SweepFactors {
		float L, M;

		SweepFactors() {
			L = 0; M = 0;
		}

		SweepFactors(float l, float m) {
			L = l; M = m;
		}
	};

	float GetNextL(float prevL, float alpha, float beta, float gamma) {
		return -gamma / (beta + alpha * prevL);
	}
	float GetNextM(float prevL, float prevM, float q, float alpha, float beta, float gamma) {
		return (q - alpha * prevM) / (beta + alpha * prevL);
	}
	float GetNewValue(float nextValue, SweepFactors lm) {
		return lm.L * nextValue + lm.M;
	}
	SweepFactors GetNextLM(SweepFactors prevLM, float q, float alpha, float beta, float gamma) {
		return SweepFactors(GetNextL(prevLM.L, alpha, beta, gamma), GetNextM(prevLM.L, prevLM.M, q, alpha, beta, gamma));
	}
};

class SweepSolveable2D : public SweepSolveable
{
protected:
	Sonsode::HostData2D<SweepFactors>* Factors;
	
	virtual bool IsZero(size_t x, size_t y) = 0;
	virtual float AlphaX(size_t x, size_t y) = 0;
	virtual float AlphaY(size_t x, size_t y) = 0;
	virtual float BetaX(size_t x, size_t y) = 0;
	virtual float BetaY(size_t x, size_t y) = 0;
	virtual float GammaX(size_t x, size_t y) = 0;
	virtual float GammaY(size_t x, size_t y) = 0;
	virtual float QX(size_t x, size_t y) = 0;
	virtual float QY(size_t x, size_t y) = 0;
	virtual float& Value(size_t x, size_t y) = 0;
	virtual size_t GetDimX() = 0;
	virtual size_t GetDimY() = 0;

public:
	void SweepX() {
		if (Factors == nullptr) return;

		for (size_t y = 1; y < GetDimY() - 1; y++) {
			//Прямая прогонка
			Factors->at(0, y).L = IsZero(0, y) ? 0.0f : 1.0f;
			Factors->at(GetDimX() - 1, y).L = IsZero(GetDimX() - 1, y) ? 0.0f : 1.0f;

			for (size_t x = 0; x < GetDimX() - 1; x++) {
				Factors->at(x + 1, y)
					= GetNextLM(Factors->at(x, y), QX(x, y), AlphaX(x, y), BetaX(x, y), GammaX(x, y));
			}
			
			//Обратная прогонка
			for (size_t x = GetDimX() - 2; x >= 1; x--)
				Value(x, y) = GetNewValue(Value(x + 1, y), Factors->at(x + 1, y));
		}
	}
	void SweepY() {
		if (Factors == nullptr) return;

		for (size_t x = 1; x < GetDimX() - 1; x++) {
			//Прямая пронка
			Factors->at(x, 0).L = IsZero(x, 0) ? 0.0f : 1.0f;
			Factors->at(x, GetDimY() - 1).L = IsZero(x, GetDimY() - 1) ? 0.0f : 1.0f;

			for (size_t y = 0; y < GetDimY() - 1; y++) {
				Factors->at(x, y + 1)
					= GetNextLM(Factors->at(x, y), QY(x, y), AlphaY(x, y), BetaY(x, y), GammaY(x, y));
			}

			//Обратная прогонка
			for (size_t y = GetDimY() - 2; y >= 1; y--)
				Value(x, y) = GetNewValue(Value(x, y + 1), Factors->at(x, y + 1));
		}
	}
	void InitSweep() {
		Factors = new Sonsode::HostData2D<SweepFactors>(GetDimX(), GetDimY());
	}
	void DeinitSweep() {
		delete Factors;
	}
};

class SweepSolveable3D : public SweepSolveable
{
protected:
	Sonsode::HostData3D<SweepFactors>* Factors;

	virtual bool IsZero(size_t x, size_t y, size_t z) = 0;
	virtual float AlphaX(size_t x, size_t y, size_t z) = 0;
	virtual float AlphaY(size_t x, size_t y, size_t z) = 0;
	virtual float AlphaZ(size_t x, size_t y, size_t z) = 0;
	virtual float BetaX(size_t x, size_t y, size_t z) = 0;
	virtual float BetaY(size_t x, size_t y, size_t z) = 0;
	virtual float BetaZ(size_t x, size_t y, size_t z) = 0;
	virtual float GammaX(size_t x, size_t y, size_t z) = 0;
	virtual float GammaY(size_t x, size_t y, size_t z) = 0;
	virtual float GammaZ(size_t x, size_t y, size_t z) = 0;
	virtual float QX(size_t x, size_t y, size_t z) = 0;
	virtual float QY(size_t x, size_t y, size_t z) = 0;
	virtual float QZ(size_t x, size_t y, size_t z) = 0;
	virtual float& Value(size_t x, size_t y, size_t z) = 0;
	virtual size_t GetDimX() = 0;
	virtual size_t GetDimY() = 0;
	virtual size_t GetDimZ() = 0;
public:
	void SweepX() {
		if (Factors == nullptr) return;

		for (size_t z = 1; z < GetDimZ() - 1; z++) {
			for (size_t y = 1; y < GetDimY() - 1; y++) {
				Factors->at(0, y, z).L = IsZero(0, y, z) ? 0.0f : 1.0f;
				Factors->at(GetDimX() - 1, y, z).L = IsZero(GetDimX() - 1, y, z) ? 0.0f : 1.0f;

				//Прямая прогонка
				for (size_t x = 0; x < GetDimX() - 1; x++) {
					Factors->at(x + 1, y, z)
						= GetNextLM(Factors->at(x, y, z), QX(x, y, z), AlphaX(x, y, z), BetaX(x, y, z), GammaX(x, y, z));
				}
				//Обратная прогонка
				for (size_t x = GetDimX() - 2; x >= 1; x--) {
					Value(x, y, z) = GetNewValue(Value(x + 1, y, z), Factors->at(x + 1, y, z));
				}
			}
		}
	}
	void SweepY() {
		if (Factors == nullptr) return;

		for (size_t z = 1; z < GetDimZ() - 1; z++) {
			for (size_t x = 1; x < GetDimX() - 1; x++) {
				Factors->at(x, 0, z).L = IsZero(x, 0, z) ? 0.0f : 1.0f;
				Factors->at(x, GetDimY() - 1, z).L = IsZero(x, GetDimY() - 1, z) ? 0.0f : 1.0f;

				//Прямая пронка
				for (size_t y = 0; y < GetDimY() - 1; y++) {
					Factors->at(x, y + 1, z)
						= GetNextLM(Factors->at(x, y, z), QY(x, y, z), AlphaY(x, y, z), BetaY(x, y, z), GammaY(x, y, z));
				}
				//Обратная прогонка
				for (size_t y = GetDimY() - 2; y >= 1; y--) {
					Value(x, y, z) = GetNewValue(Value(x, y + 1, z), Factors->at(x, y + 1, z));
				}
			}
		}
	}
	void SweepZ() {
		if (Factors == nullptr) return;

		for (size_t y = 1; y < GetDimY() - 1; y++) {
			for (size_t x = 1; x < GetDimX() - 1; x++) {
				Factors->at(x, y, 0).L = IsZero(x, y, 0) ? 0.0f : 1.0f;
				Factors->at(x, y, GetDimZ() - 1).L = IsZero(x, y, GetDimZ() - 1) ? 0.0f : 1.0f;

				//Прямая прогонка
				for (size_t z = 0; z < GetDimZ() - 1; z++) {
					Factors->at(x, y, z + 1)
						= GetNextLM(Factors->at(x, y, z), QZ(x, y, z), AlphaZ(x, y, z), BetaZ(x, y, z), GammaZ(x, y, z));
				}
				//Обратная прогонка
				for (size_t z = GetDimZ() - 2; z >= 1; z--) {
					Value(x, y, z) = GetNewValue(Value(x, y, z + 1), Factors->at(x, y, z + 1));
				}
			}
		}
	}
	void InitSweep() {
		Factors = new Sonsode::HostData3D<SweepFactors>(GetDimX(), GetDimY(), GetDimZ());
	}
	void DeinitSweep() {
		delete Factors;
	}
};