#pragma once

#include <exception>

namespace DataVisualization {
	using std::exception;

	class DataVisualizationException : public exception {
	public:
		DataVisualizationException() : exception() { }
		explicit DataVisualizationException(const char * const & what) : exception(what) { }
		DataVisualizationException(const char * const & what, int num) : exception(what, num) { }
		DataVisualizationException(const exception& e) : exception(e) { }
	};
}