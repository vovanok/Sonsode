#pragma once

#include <exception>

namespace Sonsode {
	using std::exception;

	class SonsodeException : public exception {
	public:
		SonsodeException() : exception() { }
		explicit SonsodeException(const char * const & what) : exception(what) { }
		SonsodeException(const char * const & what, int num) : exception(what, num) { }
		SonsodeException(const exception& e) : exception(e) { }
	};
}