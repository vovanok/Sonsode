#pragma once

#include <vector>
#include "IPresentable.h"
#include "Region.h"
#include "Geometry.hpp"
#include "GraphicUtils.h"
#include "HostData.hpp"

class FireKmlVisualizator : public IPresentable {
private:
	float h;
	Geometry::Rect<float> realArea;
	Geometry::Rect<float> glArea;

	void Vertex(Geometry::Point<float> point) {
		Vertex(point.x, point.y);
	}

	void Vertex(float x, float y) {
		Geometry::Point<float> p(x, y);
		p.NormalizeCoordinate(realArea, glArea);
		glVertex2f(p.x, p.y);
	}

	void DrawPolygon(Geometry::Polygon<float> polygon) {
		glBegin(GL_LINES);

		int countVerts = polygon.vertexes.size();

		Geometry::Point<float> curP, nextP;
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
	std::vector<Region> fireRegions;
	std::vector<Region> forestRegions;
	std::vector<Region> forecastRegions;

	FireKmlVisualizator(float h, Geometry::Rect<float> area, Geometry::Rect<float> glArea)
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

		////Draw grid
		//glBegin(GL_POINTS);

		//for (size_t x = 0; x < isForestField.dimX(); x++) {
		//	for (size_t y = 0; y < isForestField.dimY(); y++) {
		//		//if (fireGrid->IsFireBoundNode(curInd))
		//			//glColor3f(1, 1, 1);
		//		//else 
		//		if (isFireField(x, y) && isForestField(x, y))
		//			glColor3f(1.0f, 0.0f, 1.0f);
		//		else if (isFireField(x, y))
		//			glColor3f(1.0f, 0.0f, 0.0f);
		//		else if (isForestField(x, y))
		//			glColor3f(0.0f, 1.0f, 0.0f);
		//		else
		//			glColor3f(0.5f, 0.5f, 0.5f);

		//		Vertex(x * h, y * h);
		//	}
		//}
		//
		//glEnd();


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

			////Draw start vertex
			//glColor3f(0.9f, 0.7f, 0.0f);
			//glBegin(GL_POINTS);
			//Vertex(forecastRegion.vertexes[0]);
			//glEnd();
		}
		glLineWidth(1);
	}

	virtual void Impact(char keyCode, int button, int state, int x, int y) { }
};