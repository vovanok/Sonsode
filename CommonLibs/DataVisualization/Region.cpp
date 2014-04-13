#include "Region.h"

namespace DataVisualization {
	namespace Kml {
		bool Region::PointInSide(float x, float y) const {
			for (auto innerBounder : innerBounders)
				if (innerBounder.PointInSide(x, y))
					return false;

			if (outerBounder.PointInSide(x, y))
				return true;

			return false;
		}

		bool Region::PointInSide(Point<float> point) const {
			return PointInSide(point.x, point.y);
		}

		Rect<float> Region::GetClearanceBorders() const {
			return outerBounder.GetClearanceBorders();
		}

		void Region::Normalize(const Rect<float>& src, const Rect<float>& dst) {
			outerBounder.Normalize(src, dst);
			for (Polygon<float>& innerBounder : innerBounders)
				innerBounder.Normalize(src, dst);
		}
	}
}