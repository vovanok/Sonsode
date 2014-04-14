#pragma once

#include <vector>
#include "IPresentable.h"
#include "Region.h"
#include "Geometry.hpp"
#include "GraphicUtils.h"
#include "HostData.hpp"

using std::vector;
using DataVisualization::Geometry::Rect;
using DataVisualization::Geometry::Point;
using DataVisualization::Kml::Region;

class FireKmlVisualizator : public IPresentable {
private:
	float h;
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
		
public:
	vector<DataVisualization::Kml::Region> fireRegions;
	vector<Region> forestRegions;
	vector<Region> forecastRegions;

	FireKmlVisualizator(float h, Rect<float> area, Rect<float> glArea)
		: h(h), realArea(area), glArea(glArea) { }

	virtual void Draw() {
		glPointSize(1.0f);
		glLineWidth(2.0f);

		glColor3f(1.0f, 1.0f, 1.0f);
		
		//Draw forest regions
		for (auto forestRegion : forestRegions) {
			glColor3f(0.0f, 1.0f, 0.0f);
			DrawPolygon(forestRegion.outerBounder);
			glColor3f(0.0f, 0.5f, 0.0f);
			for (auto innerBounder : forestRegion.innerBounders)
				DrawPolygon(innerBounder);
		}

		//Draw fire bounders
		glLineStipple(1, 0x3F07);
		glEnable(GL_LINE_STIPPLE);
		glLineWidth(4);
		for (auto fireRegion : fireRegions) {
			glColor3f(1.0f, 0.0f, 0.0f);
			DrawPolygon(fireRegion.outerBounder);
			glColor3f(0.5f, 0.0f, 0.0f);
			for (auto innerBounder : fireRegion.innerBounders)
				DrawPolygon(innerBounder);
		}
		glLineWidth(1);
		glDisable(GL_LINE_STIPPLE);

		glLineWidth(4);
		//Draw forecast regions
		for (auto forecastRegion : forecastRegions) {
			glColor3f(0.0f, 0.0f, 1.0f);
			DrawPolygon(forecastRegion.outerBounder);
			glColor3f(0.0f, 0.0f, 0.5f);
			for (auto innerBounder : forecastRegion.innerBounders)
				DrawPolygon(innerBounder);
			
			glColor3f(0.0f, 1.0f, 1.0f);
			glPointSize(10.0f);
			glBegin(GL_POINTS);
			for (auto vertex : forecastRegion.outerBounder.vertexes)
				Vertex(vertex);
			glEnd();
		}
		glLineWidth(1);
	}

	virtual void Impact(char keyCode, int button, int state, int x, int y) { }
};