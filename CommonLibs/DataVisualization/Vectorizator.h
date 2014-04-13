#pragma once

#include "HostData.hpp"
#include "Geometry.hpp"

using std::vector;
using Sonsode::HostData2D;
using DataVisualization::Geometry::Point;
using DataVisualization::Geometry::Polygon;

namespace DataVisualization {
	namespace Geometry {
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
			static vector<Polygon<float>> Vectorize(HostData2D<bool> rasterField, Point<float> coordLuPoint, float h);
			static void DiscretizeAndAppend(Polygon<float> bound, Point<float> luPoint, float h, bool boundIsOuter, HostData2D<bool>& field);
		private:
			HostData2D<RasterItem> rasterWorkField;

			Vectorizator(HostData2D<bool> rasterField);
			~Vectorizator();
			vector<Polygon<size_t>> GetBorders();
			Point<size_t> GetFirstBorderNotCrossNode();
			bool IsBoundNode(Point<size_t> point);
			bool IsValue(Point<size_t> point);
			bool IsCross(Point<size_t> point);
			void CrossPoint(Point<size_t> point);
			Polygon<size_t> TracePolygonFromStartNode(Point<size_t> startNode);
			Polygon<size_t> TraceNodesFromStartNode(Point<size_t> startNode, BorderType borderType);
			Point<size_t> GetNextBorderNodeFromCurrent(Point<size_t> curNode, BorderType borderType);
		};
	}
}