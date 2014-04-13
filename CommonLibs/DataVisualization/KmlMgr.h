#pragma once

#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <tinyxml.h>
#include "StringUtils.h"
#include "Region.h"
#include "Geometry.hpp"

namespace DataVisualization {
	namespace Kml {
		using std::vector;
		using std::string;
		using std::stringstream;
		using std::exception;

		vector<Region> LoadPolygonsFromFile(const string& fileName);
		void SavePolygonsToFile(const string& fileName, const string& templateFilename, const vector<Region>& regions);
	}
}