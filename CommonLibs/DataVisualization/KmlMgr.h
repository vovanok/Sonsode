#pragma once

#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <tinyxml.h>
#include "StringUtils.h"
#include "Region.h"
#include "Geometry.hpp"

namespace KmlMgr {
	std::vector<Region> LoadPolygonsFromFile(const std::string& fileName);
	void SavePolygonsToFile(const std::string& fileName, const std::string& templateFilename,
													const std::vector<Region>& regions);
};