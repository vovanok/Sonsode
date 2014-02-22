#include "Region.h"

bool Region::PointInSide(float x, float y) const {
	for (auto innerBounder : innerBounders)
		if (innerBounder.PointInSide(x, y))
			return false;

	if (outerBounder.PointInSide(x, y))
		return true;

	return false;
}

bool Region::PointInSide(Geometry::Point<float> point) const {
	return PointInSide(point.x, point.y);
}

Geometry::Rect<float> Region::GetClearanceBorders() const {
	return outerBounder.GetClearanceBorders();
}

void Region::Normalize(const Geometry::Rect<float>& src, const Geometry::Rect<float>& dst) {
	outerBounder.Normalize(src, dst);
	for (Geometry::Polygon<float>& innerBounder : innerBounders)
		innerBounder.Normalize(src, dst);
}