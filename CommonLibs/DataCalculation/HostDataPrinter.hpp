#pragma once

#include <string>
#include <sstream>
#include <functional>
#include <iomanip>
#include "HostData.hpp"

class HostDataPrinter {
public:
	template <class T> static std::string Print(const T &value);
	template <class T> static std::string Print(const Sonsode::HostData3D<T> &data);
	template <class T> static std::string Print(const Sonsode::HostData3D<T> &data, const std::function<std::string(const T&)> &customPrinter);
	template <class T> static std::string Print(const Sonsode::HostData2D<T> &data);
};

#pragma region Implementation

template<class T>
std::string HostDataPrinter::Print(const T &value) {
	std::stringstream ss;
	ss << std::fixed << std::setprecision(2) << value;
	return ss.str();
}

template<class T>
std::string HostDataPrinter::Print(const Sonsode::HostData3D<T> &data) {
	return HostDataPrinter::Print(data, Print);
}

template<class T>
std::string HostDataPrinter::Print(const Sonsode::HostData3D<T> &data, const std::function<std::string(const T&)> &customPrinter) {
	std::stringstream ss;
	for (size_t z = 0; z < data.dimZ(); z++) {
		ss << std::endl << "--------- z = " << z << " ---------" << std::endl;
		for (size_t y = 0; y < data.dimY(); y++) {
			for (size_t x = 0; x < data.dimX(); x++) {
				ss << customPrinter(data.at(x, y, z));
				if (x != data.dimX() - 1)
					ss << " ";
			}
			ss << std::endl;
		}
	}
	return ss.str();
}
	
template<class T>
std::string HostDataPrinter::Print(const Sonsode::HostData2D<T> &data) {
	std::stringstream ss;
	for (size_t y = 0; y < data.dimY(); y++) {
		for (size_t x = 0; x < data.dimX(); x++) {
			ss << Print<T>(data.at(x, y));
			if (x != data.dimX() - 1)
				ss << " ";
		}
		ss << std::endl;
	}
	return ss.str();
}

#pragma endregion