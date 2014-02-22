#ifndef STRINGUTILS_H
#define STRINGUTILS_H

#include <string>
#include <vector>
#include <algorithm>

#define EMPTY_STRING ""

namespace StringUtils
{
	std::vector<std::string> SplitString(std::string strValue, std::string splitter);
	float StringToFloat(std::string value);
	bool StringToBool(std::string value);
};

#endif