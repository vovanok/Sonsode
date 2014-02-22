#pragma once

#include "SonsodeCommon.h"

namespace Sonsode {
	template<class T>
	class HostData {
	public:
		virtual void Fill(T value) = 0;
		void Erase() const { delete[] _data; }
		T *data() const { return _data; }

	protected:
		T *_data;
	};

	template<class T>
	class HostData1D : public HostData<T> {
	public:
		HostData1D() : _dimX(0) { }
		explicit HostData1D(size_t dimX) : _dimX(dimX) {
			_data = new T[dimX];
		}

		void Fill(T value) {
			for (size_t i = 0; i < dimX(); i++)
				_data[i] = value;
		}

		T& at(size_t x) const { return this->operator()(x); }
		T& operator()(size_t x) const { return _data[(x <= dimX() - 1) * x]; }
		size_t dimX() const { return _dimX; }

	protected:
		size_t _dimX;
	};

	template<class T>
	class HostData2D : public HostData<T> {
	public:
		HostData2D() : _dimX(0), _dimY(0) { }
		HostData2D(size_t dimX, size_t dimY)
				: _dimX(dimX), _dimY(dimY) {
			_data = new T[dimX * dimY];
		}

		void Fill(T value) {
			for (size_t i = 0; i < dimX() * dimY(); i++)
			_data[i] = value;
		}

		T& at(size_t x, size_t y) const { return this->operator()(x, y); }
		T& operator()(size_t x, size_t y) const {
			return _data[(x <= dimX() - 1 && y <= dimY() - 1) * (y * dimX() + x)];
		}

		size_t dimX() const { return _dimX; }
		size_t dimY() const { return _dimY; }

	protected:
		size_t _dimX;
		size_t _dimY;
	};

	template<class T>
	class HostData3D : public HostData<T> {
	public:
		HostData3D() : _dimX(0), _dimY(0), _dimZ(0) { }
		HostData3D(size_t dimX, size_t dimY, size_t dimZ)
				: _dimX(dimX), _dimY(dimY), _dimZ(dimZ) {
			_data = new T[dimX * dimY * dimZ];
		}

		void Fill(T value) {
			for (size_t i = 0; i < dimX() * dimY() * dimZ(); i++)
			_data[i] = value;
		}

		T& at(size_t x, size_t y, size_t z) const { return this->operator()(x, y, z); }
		T& operator()(size_t x, size_t y, size_t z) const {
			return _data[(x <= dimX() - 1 && y <= dimY() - 1 && z <= dimZ() - 1) * (z * dimX() * dimY() + y * dimX() + x)];
		}

		size_t dimX() const { return _dimX; }
		size_t dimY() const { return _dimY; }
		size_t dimZ() const { return _dimZ; }

	protected:
		size_t _dimX;
		size_t _dimY;
		size_t _dimZ;
	};

	template<class T>
	class HostData4D : public HostData<T> {
	public:
		HostData4D() : _dimX(0), _dimY(0), _dimZ(0), _dimW(0) { }
		HostData4D(size_t dimX, size_t dimY, size_t dimZ, size_t dimW)
				: _dimX(dimX), _dimY(dimY), _dimZ(dimZ), _dimW(dimW) {
			_data = new T[dimX * dimY * dimZ * dimW];
		}

		void Fill(T value) {
			for (size_t i = 0; i < dimX() * dimY() * dimZ() * dimW(); i++)
				_data[i] = value;
		}

		T& at(size_t x, size_t y, size_t z, size_t w) const { return this->operator()(x, y, z, w); }
		T& operator()(size_t x, size_t y, size_t z, size_t w) const {
			return _data[(x < dimX() && y < dimY() && z < dimZ() && w < dimW()) * (w * dimX() * dimY() * dimZ() + z * dimX() * dimY() + y * dimX() + x)];
		}

		size_t dimX() const { return _dimX; }
		size_t dimY() const { return _dimY; }
		size_t dimZ() const { return _dimZ; }
		size_t dimW() const { return _dimW; }

	protected:
		size_t _dimX;
		size_t _dimY;
		size_t _dimZ;
		size_t _dimW;
	};
};