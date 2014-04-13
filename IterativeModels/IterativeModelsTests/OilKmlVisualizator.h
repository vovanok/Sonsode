#pragma once

#include <vector>
#include "IPresentable.h"
#include "Region.h"
#include "Geometry.hpp"
#include "GraphicUtils.h"
#include "HostData.hpp"
#include "ColorPalette.h"

using DataVisualization::Kml::Region;
using DataVisualization::Geometry::Point;
using DataVisualization::Geometry::Rect;
using DataVisualization::Geometry::Polygon;

struct LineDescriptor {
public:
	Color color;
	float width;
	int factor;
	unsigned short pattern;

	LineDescriptor()
		: color(Color(1.0f, 1.0f, 1.0f, 1.0f)), width(1.0f),
			factor(1), pattern(0x1111), isLineStipple(false) { }
	LineDescriptor(Color color, float width)
		: color(color), width(width), factor(1),
			pattern(0x1111), isLineStipple(false) { }
	LineDescriptor(Color color, float width, int factor, unsigned short pattern)
		: color(color), width(width), factor(factor),
			pattern(pattern), isLineStipple(true) { }

	void Apply() const {
		glLineWidth(width);
		glColor4f(color.r, color.g, color.b, color.a);
		if (isLineStipple) {
			glLineStipple(factor, pattern);
			glEnable(GL_LINE_STIPPLE);
		} else {
			glDisable(GL_LINE_STIPPLE);
		}
	}

private:
	bool isLineStipple;
};

class OilKmlVisualizator : public IPresentable {
private:
	Point<float> h;
	Rect<float> realArea;
	Rect<float> glArea;

	void Vertex(Point<float> point) {
		Vertex(point.x, point.y);
	}

	void Vertex(float x, float y) {
		Point<float> p(x, y);
		p.NormalizeCoordinate(realArea, glArea);
		glVertex2f(p.x, p.y);
	}

	void DrawPolygon(DataVisualization::Geometry::Polygon<float> polygon) {
		glBegin(GL_LINES);

		int countVerts = polygon.vertexes.size();

		Point<float> curP, nextP;
		for (int curPointInd = 0; curPointInd < countVerts - 1; curPointInd++) {
			curP = polygon.vertexes[curPointInd];
			nextP = polygon.vertexes[curPointInd + 1];

			Vertex(curP);
			Vertex(nextP);
		}

		Vertex(polygon.vertexes[countVerts - 1]);
		Vertex(polygon.vertexes[0]);

		glEnd();
	}
	
	void DrawRegion(const Region& region, const LineDescriptor& outerLine, const LineDescriptor& innerLine) {
		outerLine.Apply();
		DrawPolygon(region.outerBounder);
		innerLine.Apply();
		for (auto innerBounder : region.innerBounders)
			DrawPolygon(innerBounder);
	}

public:
	vector<Region> waterRegions;
	vector<Region> oilRegions;
	vector<Region> forecastRegions;

	Sonsode::HostData2D<bool> isWaterField;
	Sonsode::HostData2D<bool> isOilField;
	Sonsode::HostData2D<bool> isForecastField;

	OilKmlVisualizator(Point<float> h, Rect<float> area, Rect<float> glArea)
		: h(h), realArea(area), glArea(glArea) { }

	virtual void Draw() {
		//Draw water regions
		for (auto waterRegion : waterRegions)
			DrawRegion(waterRegion,
								 LineDescriptor(Color(0.0f, 0.0f, 1.0f, 1.0f), 2.0f),
								 LineDescriptor(Color(0.0f, 0.0f, 0.9f, 1.0f), 2.0f));

		//Draw oil bounders
		for (auto oilRegion : oilRegions)
			DrawRegion(oilRegion,
								 LineDescriptor(Color(0.4f, 0.4f, 0.4f, 1.0f), 3.0f, 1, 0x3F07),
								 LineDescriptor(Color(0.4f, 0.4f, 0.4f, 1.0f), 3.0f, 1, 0x3F07));

		//Draw forecast regions
		for (auto forecastRegion : forecastRegions) {
			DrawRegion(forecastRegion,
								 LineDescriptor(Color(0.0f, 0.0f, 0.0f, 1.0f), 4.0f),
								 LineDescriptor(Color(0.0f, 0.0f, 0.0f, 1.0f), 4.0f));

			glColor3f(0.0f, 1.0f, 1.0f);
			glPointSize(10.0f);
			glBegin(GL_POINTS);
			for (auto vertex : forecastRegion.outerBounder.vertexes)
				Vertex(vertex);
			glEnd();
		}

		//Draw grid
		glPointSize(2.0f);
		glBegin(GL_POINTS);
		
		for (size_t x = 0; x < isWaterField.dimX(); x++) {
			for (size_t y = 0; y < isWaterField.dimY(); y++) {
				if (isWaterField(x, y))
					glColor3f(0.2f, 0.2f, 1.0f);
				else
					glColor3f(0.6f, 1.0f, 0.6f);

				if (isOilField(x, y))
					glColor3f(0.0f, 0.0f, 0.0f);

				if (isForecastField(x, y))
					glColor3f(1.0f, 0.0f, 0.0f);
		
				Vertex(x * h.x + realArea.point1.x, y * h.y + realArea.point1.y);
			}
		}
		
		glEnd();

		glLineWidth(1);
	}

	virtual void Impact(char keyCode, int button, int state, int x, int y) { }
};