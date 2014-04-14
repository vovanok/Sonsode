#include "StringUtils.h"

std::vector<std::string> StringUtils::SplitString(std::string strValue, std::string splitter) {
	std::vector<std::string> parts(0);
	if (strValue == "")
		return parts;

	size_t curCursorPosition = 0;
	size_t newCursorPosition = 1;
	while ((newCursorPosition != std::string::npos) && (curCursorPosition < strValue.length())) {
		newCursorPosition = strValue.find_first_of(splitter, curCursorPosition);
		std::string part = (newCursorPosition != std::string::npos)
			? strValue.substr(curCursorPosition, newCursorPosition - curCursorPosition)
			: strValue.substr(curCursorPosition, strValue.length() - curCursorPosition);
		curCursorPosition = newCursorPosition + splitter.length();
		parts.push_back(part);
	}
	return parts;
}

float StringUtils::StringToFloat(std::string value) {
	std::size_t dotPosition = value.find_first_of(".");
	if (dotPosition != std::string::npos)
		value = value.replace(dotPosition, 1, ",");
		
	return std::stof(value);
}

bool StringUtils::StringToBool(std::string value) {
	if (value == EMPTY_STRING)
		return false;

	std::string checkingValue = value;
	std::transform(checkingValue.begin(), checkingValue.end(), checkingValue.begin(), ::tolower);

	if (checkingValue == "true")
		return true;
	if (checkingValue == "false")
		return false;
	
	return checkingValue != "0";
}