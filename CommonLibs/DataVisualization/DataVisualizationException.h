#pragma once

#include <exception>

class DataVisualizationException : public std::exception {
public:
	DataVisualizationException() : std::exception() { }
	explicit DataVisualizationException(const char * const & what) : std::exception(what) { }
	DataVisualizationException(const char * const & what, int num) : std::exception(what, num) { }
	DataVisualizationException(const exception& e) : std::exception(e) { }
};