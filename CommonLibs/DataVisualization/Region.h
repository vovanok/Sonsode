#pragma once

#include <vector>
#include "Geometry.hpp"

class Region {
public:
	Geometry::Polygon<float> outerBounder;
	std::vector<Geometry::Polygon<float>> innerBounders;

	Region(Geometry::Polygon<float> outerBounder, std::vector<Geometry::Polygon<float>> innerBounders)
		: outerBounder(outerBounder), innerBounders(innerBounders) { }

	bool PointInSide(float x, float y) const;
	bool PointInSide(Geometry::Point<float> point) const;
	Geometry::Rect<float> GetClearanceBorders() const;
	void Normalize(const Geometry::Rect<float>& src, const Geometry::Rect<float>& dst);
};