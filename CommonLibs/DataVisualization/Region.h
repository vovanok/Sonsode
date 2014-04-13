#pragma once

#include <vector>
#include "Geometry.hpp"

namespace DataVisualization {
	namespace Kml {
		using std::vector;
		using DataVisualization::Geometry::Polygon;
		using DataVisualization::Geometry::Point;
		using DataVisualization::Geometry::Rect;
		
		class Region {
		public:
			Polygon<float> outerBounder;
			vector<Polygon<float>> innerBounders;

			Region(Polygon<float> outerBounder, vector<Polygon<float>> innerBounders)
				: outerBounder(outerBounder), innerBounders(innerBounders) { }

			bool PointInSide(float x, float y) const;
			bool PointInSide(Point<float> point) const;
			Rect<float> GetClearanceBorders() const;
			void Normalize(const Rect<float>& src, const Rect<float>& dst);
		};
	}
}