#pragma once

#include "HostData.hpp"
#include "Geometry.hpp"

using Sonsode::HostData2D;

enum class BorderType {
	Left,
	Right
};

class RasterItem {
public:
	bool isValue;
	bool isCross;

	RasterItem() : isValue(false), isCross(false) { }
	explicit RasterItem(bool isValue) : isValue(isValue), isCross(false) { }
	RasterItem(bool isValue, bool isCross) : isValue(isValue), isCross(isCross) { }
};

class Vectorizator {
public:
	static std::vector<Geometry::Polygon<float>> Vectorize(HostData2D<bool> rasterField,
																												 Geometry::Point<float> coordLuPoint, float h);
	static void DiscretizeAndAppend(Geometry::Polygon<float> bound, Geometry::Point<float> luPoint,
																	float h, bool boundIsOuter, HostData2D<bool>& field);
private:
	HostData2D<RasterItem> rasterWorkField;

  Vectorizator(HostData2D<bool> rasterField);
	~Vectorizator();
	std::vector<Geometry::Polygon<size_t>> GetBorders();
  Geometry::Point<size_t> GetFirstBorderNotCrossNode();
  bool IsBoundNode(Geometry::Point<size_t> point);
  bool IsValue(Geometry::Point<size_t> point);
  bool IsCross(Geometry::Point<size_t> point);
  void CrossPoint(Geometry::Point<size_t> point);
  Geometry::Polygon<size_t> TracePolygonFromStartNode(Geometry::Point<size_t> startNode);
  Geometry::Polygon<size_t> TraceNodesFromStartNode(Geometry::Point<size_t> startNode, BorderType borderType);
  Geometry::Point<size_t> GetNextBorderNodeFromCurrent(Geometry::Point<size_t> curNode, BorderType borderType);
};