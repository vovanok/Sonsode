#include "KmlMgr.h"
#include "DataVisualizationException.h"

namespace DataVisualization {
	namespace Kml {

		namespace {
			const string KML_ELNAME = "kml";
			const string DOCUMENT_ELNAME = "Document";
			const string PLACEMARK_ELNAME = "Placemark";
			const string POLYGON_ELNAME = "Polygon";
			const string OUTERBOUNDARYIS_ELNAME = "outerBoundaryIs";
			const string INNERBOUNDARYIS_ELNAME = "innerBoundaryIs";
			const string COORDINATES_ELNAME = "coordinates";
			const string LINEARRING_ELNAME = "LinearRing";
			const string MULTIGEOMETRY_ELNAME = "MultiGeometry";

			vector<TiXmlElement*> FindAllElemsIntoOwner(TiXmlElement* owner, const string& name) {
				vector<TiXmlElement*> result(0);
	
				if (owner == nullptr)
					return result;

				for (TiXmlElement* curEl = owner->FirstChildElement(); curEl; curEl = curEl->NextSiblingElement()) {
		
					if (string(curEl->Value()) == name)
						result.push_back(curEl);

					vector<TiXmlElement*> curRes = FindAllElemsIntoOwner(curEl, name);
					result.insert(result.end(), curRes.begin(), curRes.end());
				}
				return result;
			}
	
			Point<float> StringToPoint(const string& strValue) {
				vector<string> curCoordsStr = StringUtils::SplitString(strValue, ",");
				if (curCoordsStr.size() >= 2) {
					float x = StringUtils::StringToFloat(curCoordsStr[0]);
					float y = StringUtils::StringToFloat(curCoordsStr[1]);
					return Point<float>(x, y);
				}

				return Point<float>(0.0f, 0.0f);
			}

			string PointToString(const Point<float>& pointValue) {
				stringstream ss;
				ss << pointValue.x << "," << pointValue.y << ",0";
				return ss.str();
			}

			vector<Point<float>> StringToPoints(const string& strValue) {
				vector<Point<float>> points(0);
				if (strValue == "")
					return points;
			
				vector<string> coordsStr = StringUtils::SplitString(strValue, " ");

				for (size_t coordStrNum = 0; coordStrNum < coordsStr.size(); coordStrNum++) {
					if (coordsStr[coordStrNum] != "")
						points.push_back(StringToPoint(coordsStr[coordStrNum]));
				}

				return points;
			}

			string PointsToString(const vector<Point<float>>& pointsValue) {
				stringstream ss;
				for (size_t pointNum = 0; pointNum < pointsValue.size(); pointNum++) {
					ss << PointToString(pointsValue[pointNum]);
					if (pointNum < pointsValue.size() - 1)
						ss << " ";
				}
				return ss.str();
			}

			Polygon<float> GetPolygonByBoundaryInfo(TiXmlElement* boundaryInfo) {
				if (boundaryInfo == nullptr)
					return Polygon<float>();

				vector<TiXmlElement*> coordElems =
					FindAllElemsIntoOwner(boundaryInfo, COORDINATES_ELNAME);

				if (coordElems.size() == 0)
					return Polygon<float>();

				string coordinates(coordElems[0]->GetText());
				vector<Point<float>> points = StringToPoints(coordinates);

				return Polygon<float>(points);
			}

			TiXmlElement* GetCoordinatesKmlElementByPolygon(const Polygon<float>& polygon) {
				TiXmlElement* result = new TiXmlElement(COORDINATES_ELNAME.c_str());
				TiXmlText* text = new TiXmlText(PointsToString(polygon.vertexes).c_str());
				result->LinkEndChild(text);
				return result;
			}

			TiXmlElement* GetBoundaryIs(const Polygon<float>& polygon, bool isOuter) {
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

		vector<Region> LoadPolygonsFromFile(const string& fileName) {
			TiXmlDocument document(fileName.c_str());
			if (!document.LoadFile())
				throw DataVisualizationException("Ошибка доступа к kml");

			vector<Region> regions;

			try {
				TiXmlHandle docHandle(&document);

				TiXmlElement* placemarkEl = docHandle
					.FirstChild(KML_ELNAME.c_str())
					.FirstChild(DOCUMENT_ELNAME.c_str())
					.FirstChild(PLACEMARK_ELNAME.c_str()).ToElement();

				vector<TiXmlElement*> polygonElems = FindAllElemsIntoOwner(placemarkEl, POLYGON_ELNAME);

				//Интерпретация каждого Polygon в Region
				for (auto polygonElem : polygonElems) {
					//Внешние границы
					vector<TiXmlElement*> outerBoundersElems = FindAllElemsIntoOwner(polygonElem, OUTERBOUNDARYIS_ELNAME);
					if (outerBoundersElems.size() == 0)
						continue;

					Polygon<float> outerBoundaryPolygon = GetPolygonByBoundaryInfo(outerBoundersElems[0]);
					if (outerBoundaryPolygon.vertexes.size() == 0)
						continue;

					//Внутренние границы
					vector<TiXmlElement*> innerBoundersElems = FindAllElemsIntoOwner(polygonElem, INNERBOUNDARYIS_ELNAME);

					vector<Polygon<float>> innerBoundaryPolygons(0);
					for (auto boundElem : innerBoundersElems) {
						Polygon<float> innerBoundaryPolygon = GetPolygonByBoundaryInfo(boundElem);
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

		void SavePolygonsToFile(const string& fileName, const string& templateFilename, const vector<Region>& regions) {
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

				vector<TiXmlElement*> placemarkEls = FindAllElemsIntoOwner(kmlEl, PLACEMARK_ELNAME);
				if (placemarkEls.size() == 0)
					throw DataVisualizationException("Шаблон файла KML некорректен");

				placemarkEls[0]->InsertEndChild(*mount);
				document.SaveFile(fileName.c_str());
			} catch(std::exception e) {
				throw DataVisualizationException(e);
			}
		}
	}
}