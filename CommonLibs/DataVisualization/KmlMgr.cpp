#include "KmlMgr.h"
#include "DataVisualizationException.h"

namespace KmlMgr {
	namespace {
		const std::string KML_ELNAME = "kml";
		const std::string DOCUMENT_ELNAME = "Document";
		const std::string PLACEMARK_ELNAME = "Placemark";
		const std::string POLYGON_ELNAME = "Polygon";
		const std::string OUTERBOUNDARYIS_ELNAME = "outerBoundaryIs";
		const std::string INNERBOUNDARYIS_ELNAME = "innerBoundaryIs";
		const std::string COORDINATES_ELNAME = "coordinates";
		const std::string LINEARRING_ELNAME = "LinearRing";
		const std::string MULTIGEOMETRY_ELNAME = "MultiGeometry";

		std::vector<TiXmlElement*> FindAllElemsIntoOwner(TiXmlElement* owner, const std::string& name) {
			std::vector<TiXmlElement*> result(0);
	
			if (owner == nullptr)
				return result;

			for (TiXmlElement* curEl = owner->FirstChildElement(); curEl; curEl = curEl->NextSiblingElement()) {
		
				if (std::string(curEl->Value()) == name)
					result.push_back(curEl);

				std::vector<TiXmlElement*> curRes = FindAllElemsIntoOwner(curEl, name);
				result.insert(result.end(), curRes.begin(), curRes.end());
			}
			return result;
		}
	
		Geometry::Point<float> StringToPoint(const std::string& strValue) {
			std::vector<std::string> curCoordsStr = StringUtils::SplitString(strValue, ",");
			if (curCoordsStr.size() >= 2) {
				float x = StringUtils::StringToFloat(curCoordsStr[0]);
				float y = StringUtils::StringToFloat(curCoordsStr[1]);
				return Geometry::Point<float>(x, y);
			}

			return Geometry::Point<float>(0.0f, 0.0f);
		}

		std::string PointToString(const Geometry::Point<float>& pointValue) {
			std::stringstream ss;
			ss << pointValue.x << "," << pointValue.y << ",0";
			return ss.str();
		}

		std::vector<Geometry::Point<float>> StringToPoints(const std::string& strValue) {
			std::vector<Geometry::Point<float>> points(0);
			if (strValue == "")
				return points;
			
			std::vector<std::string> coordsStr = StringUtils::SplitString(strValue, " ");

			for (size_t coordStrNum = 0; coordStrNum < coordsStr.size(); coordStrNum++) {
				if (coordsStr[coordStrNum] != "")
					points.push_back(StringToPoint(coordsStr[coordStrNum]));
			}

			return points;
		}

		std::string PointsToString(const std::vector<Geometry::Point<float>>& pointsValue) {
			std::stringstream ss;
			for (size_t pointNum = 0; pointNum < pointsValue.size(); pointNum++) {
				ss << PointToString(pointsValue[pointNum]);
				if (pointNum < pointsValue.size() - 1)
					ss << " ";
			}
			return ss.str();
		}

		Geometry::Polygon<float> GetPolygonByBoundaryInfo(TiXmlElement* boundaryInfo) {
			if (boundaryInfo == nullptr)
				return Geometry::Polygon<float>();

			std::vector<TiXmlElement*> coordElems =
				FindAllElemsIntoOwner(boundaryInfo, COORDINATES_ELNAME);

			if (coordElems.size() == 0)
				return Geometry::Polygon<float>();

			std::string coordinates(coordElems[0]->GetText());
			std::vector<Geometry::Point<float>> points = StringToPoints(coordinates);

			return Geometry::Polygon<float>(points);
		}

		TiXmlElement* GetCoordinatesKmlElementByPolygon(const Geometry::Polygon<float>& polygon) {
			TiXmlElement* result = new TiXmlElement(COORDINATES_ELNAME.c_str());
			TiXmlText* text = new TiXmlText(PointsToString(polygon.vertexes).c_str());
			result->LinkEndChild(text);
			return result;
		}

		TiXmlElement* GetBoundaryIs(const Geometry::Polygon<float>& polygon, bool isOuter) {
			TiXmlElement* coordinates =
				GetCoordinatesKmlElementByPolygon(polygon);

			TiXmlElement* linearRing = new TiXmlElement(LINEARRING_ELNAME.c_str());
			linearRing->LinkEndChild(coordinates);

			TiXmlElement* result = new TiXmlElement(
				isOuter ? OUTERBOUNDARYIS_ELNAME.c_str() : INNERBOUNDARYIS_ELNAME.c_str());
			result->LinkEndChild(linearRing);

			return result;
		}

		TiXmlElement* GetPolygonKmlElementByRegion(const Region* region) {
			TiXmlElement* outerBoundaryIs = GetBoundaryIs(region->outerBounder, true);
		
			TiXmlElement* resultPolygon = new TiXmlElement(POLYGON_ELNAME.c_str());
			resultPolygon->InsertEndChild(*outerBoundaryIs);

			for (size_t innerBoundNum = 0; innerBoundNum < region->innerBounders.size(); innerBoundNum++) 	{
				TiXmlElement* innerBoundaryIs = GetBoundaryIs(region->innerBounders[innerBoundNum], false);
				resultPolygon->InsertEndChild(*innerBoundaryIs);
			}

			return resultPolygon;
		}
	}

	std::vector<Region> LoadPolygonsFromFile(const std::string& fileName) {
		TiXmlDocument document(fileName.c_str());
		if (!document.LoadFile())
			throw DataVisualizationException("Ошибка доступа к kml");

		std::vector<Region> regions;

		try {
			TiXmlHandle docHandle(&document);

			TiXmlElement* placemarkEl = docHandle
				.FirstChild(KML_ELNAME.c_str())
				.FirstChild(DOCUMENT_ELNAME.c_str())
				.FirstChild(PLACEMARK_ELNAME.c_str()).ToElement();

			std::vector<TiXmlElement*> polygonElems =
				FindAllElemsIntoOwner(placemarkEl, POLYGON_ELNAME);

			//Интерпретация каждого Polygon в Region
			for (auto polygonElem : polygonElems) {
				
				//Внешние границы
				std::vector<TiXmlElement*> outerBoundersElems = 
					FindAllElemsIntoOwner(polygonElem, OUTERBOUNDARYIS_ELNAME);

				if (outerBoundersElems.size() == 0)
					continue;

				Geometry::Polygon<float> outerBoundaryPolygon =
					GetPolygonByBoundaryInfo(outerBoundersElems[0]);

				if (outerBoundaryPolygon.vertexes.size() == 0)
					continue;

				//Внутренние границы
				std::vector<TiXmlElement*> innerBoundersElems = 
					FindAllElemsIntoOwner(polygonElem, INNERBOUNDARYIS_ELNAME);

				std::vector<Geometry::Polygon<float>> innerBoundaryPolygons(0);
				for (auto boundElem : innerBoundersElems) {
					Geometry::Polygon<float> innerBoundaryPolygon =
						GetPolygonByBoundaryInfo(boundElem);
					
					if (innerBoundaryPolygon.vertexes.size() != 0)
						innerBoundaryPolygons.push_back(innerBoundaryPolygon);
				}

				regions.push_back(Region(outerBoundaryPolygon, innerBoundaryPolygons));
			}
		} catch(std::exception e) {
			throw DataVisualizationException(e);
		}

		return regions;
	}

	void SavePolygonsToFile(const std::string& fileName, const std::string& templateFilename,
													const std::vector<Region>& regions) {
		try {
			if (regions.size() == 0)
				return;

			TiXmlDocument document(templateFilename.c_str());
			if (!document.LoadFile())
				throw DataVisualizationException("Ошибка загрузки шаблона файла KML");
		
			TiXmlElement* mount;
			if (regions.size() == 1) {
				mount = GetPolygonKmlElementByRegion(&regions[0]);
			} else {
				mount = new TiXmlElement(MULTIGEOMETRY_ELNAME.c_str());
				for (auto region : regions)
					mount->InsertEndChild(*GetPolygonKmlElementByRegion(&region));
			}

			TiXmlHandle docHandle(&document);
			TiXmlElement* kmlEl = docHandle.FirstChild(KML_ELNAME.c_str()).ToElement();
			if (!kmlEl)
				throw DataVisualizationException("Шаблон файла KML некорректен");

			std::vector<TiXmlElement*> placemarkEls = FindAllElemsIntoOwner(kmlEl, PLACEMARK_ELNAME);
			if (placemarkEls.size() == 0)
				throw DataVisualizationException("Шаблон файла KML некорректен");

			placemarkEls[0]->InsertEndChild(*mount);
			document.SaveFile(fileName.c_str());
		} catch (std::exception e) {
			throw DataVisualizationException(e);
		}
	}
}