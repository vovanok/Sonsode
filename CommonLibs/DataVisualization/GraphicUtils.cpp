#include "GraphicUtils.h"

namespace DataVisualization {
	namespace Graphic {
		namespace {
			void GlColor(Color color) {
				glColor4f(color.r, color.g, color.b, color.a);
			}

			void DrawOneVector(const Vector3D<size_t>& gridCoords, const Vector3D<float>& vectorValue,
												 const Grid3DCoordSys& cs, float minValue, float normLimit, float normFactor) {
				if (vectorValue.x <= minValue && vectorValue.y <= minValue && vectorValue.z <= minValue)
						return;

				Vector3D<float> beginPoint = cs.GetGrid3DCoords(gridCoords);
				Vector3D<float> delta = Vector3D<float>(
					vectorValue.x * normFactor, vectorValue.y * normFactor, vectorValue.z * normFactor);

				delta.x = delta.x > normLimit ? normLimit : (delta.x < - normLimit ? - normLimit : delta.x);
				delta.y = delta.y > normLimit ? normLimit : (delta.y < - normLimit ? - normLimit : delta.y);
				delta.z = delta.z > normLimit ? normLimit : (delta.z < - normLimit ? - normLimit : delta.z);

				Vector3D<float> endPoint = Vector3D<float>(
					beginPoint.x + delta.x, beginPoint.y + delta.y, beginPoint.z + delta.z);

				glBegin(GL_LINES);
				cs.Vertex(beginPoint);
				cs.Vertex(endPoint);
				glEnd();

				glBegin(GL_POINTS);
				cs.Vertex(beginPoint);
				glEnd();
			}
		}

		void DrawColoredSpace(const HostData3D<float>& values, const Vector3D<size_t>& curPlaneNum,
													const ColorPalette& cp, const Grid3DCoordSys& cs) {
			//Отрисовка поверхности перпендикулярной X
			glBegin(GL_TRIANGLES);
			if (values.dimY() >= 2 && values.dimZ() >= 2) {
				for (size_t y = 0; y <= values.dimY()-2; y++) {
					for (size_t z = 0; z <= values.dimZ()-2; z++) {
						GlColor(cp.GetColor(values(curPlaneNum.x, y, z)));
						cs.Vertex(curPlaneNum.x, y, z);
						GlColor(cp.GetColor(values(curPlaneNum.x, y, z + 1)));
						cs.Vertex(curPlaneNum.x, y, z + 1);
						GlColor(cp.GetColor(values(curPlaneNum.x, y + 1, z)));
						cs.Vertex(curPlaneNum.x, y + 1, z);

						cs.Vertex(curPlaneNum.x, y + 1, z);
						GlColor(cp.GetColor(values(curPlaneNum.x, y, z + 1)));
						cs.Vertex(curPlaneNum.x, y, z + 1);
						GlColor(cp.GetColor(values(curPlaneNum.x, y + 1, z + 1)));
						cs.Vertex(curPlaneNum.x, y + 1, z + 1);
					}
				}
			}

			if (values.dimZ() >= 2 && values.dimX() >= 2) {
				for (size_t z = 0; z <= values.dimZ()-2; z++) {
					for (size_t x = 0; x <= values.dimX()-2; x++) {
						GlColor(cp.GetColor(values(x, curPlaneNum.y, z)));
						cs.Vertex(x, curPlaneNum.y, z);
						GlColor(cp.GetColor(values(x, curPlaneNum.y, z+1)));
						cs.Vertex(x, curPlaneNum.y, z+1);
						GlColor(cp.GetColor(values(x+1, curPlaneNum.y, z)));
						cs.Vertex(x+1, curPlaneNum.y, z);

						cs.Vertex(x+1, curPlaneNum.y, z);
						GlColor(cp.GetColor(values(x, curPlaneNum.y, z+1)));
						cs.Vertex(x, curPlaneNum.y, z+1);
						GlColor(cp.GetColor(values(x+1, curPlaneNum.y, z+1)));
						cs.Vertex(x+1, curPlaneNum.y, z+1);
					}
				}
			}

			if (values.dimY() >= 2 && values.dimX() >= 2) {
				for (size_t y = 0; y <= values.dimY()-2; y++) {
					for (size_t x = 0; x <= values.dimX()-2; x++) {
						GlColor(cp.GetColor(values(x, y, curPlaneNum.z)));
						cs.Vertex(x, y, curPlaneNum.z);
						GlColor(cp.GetColor(values(x, y+1, curPlaneNum.z)));
						cs.Vertex(x, y+1, curPlaneNum.z);
						GlColor(cp.GetColor(values(x+1, y, curPlaneNum.z)));
						cs.Vertex(x+1, y, curPlaneNum.z);

						cs.Vertex(x+1, y, curPlaneNum.z);
						GlColor(cp.GetColor(values(x, y+1, curPlaneNum.z)));
						cs.Vertex(x, y+1, curPlaneNum.z);
						GlColor(cp.GetColor(values(x+1, y+1, curPlaneNum.z)));
						cs.Vertex(x+1, y+1, curPlaneNum.z);
					}
				}
			}
			glEnd();		

			//Отрисовка реберного куба
			glColor3f(0.0f, 0.4f, 0.0f);
			DrawCostalCude(
				cs.GetGlCoords(Vector3D<size_t>(0, 0, 0)), 
				cs.GetGlCoords(Vector3D<size_t>(values.dimX()-1, values.dimY()-1, values.dimZ()-1)));
		}

		void DrawDissectedVectorSpace(const HostData3D<Vector3D<float>>& vectors, float minValue, float maxValue,
																	const Vector3D<size_t>& curPlaneNum, const Grid3DCoordSys& cs) {
			float normLimit = cs.H();
			float normFactor = normLimit / maxValue;

			glPointSize(2);

			glColor3f(0.0f, 0.0f, 0.0f);
			//Перпендикулярно X
			for (size_t z = 0; z <= vectors.dimZ()-1; z++) {
				for (size_t y = 0; y <= vectors.dimY()-1; y++) {
					DrawOneVector(Vector3D<size_t>(curPlaneNum.x, y, z),
						vectors.at(curPlaneNum.x, y, z), cs, minValue, normLimit, normFactor);
				}
			}

			//Перпендикулярно Y
			for (size_t z = 0; z <= vectors.dimZ()-1; z++) {
				for (size_t x = 0; x <= vectors.dimX()-1; x++) {
					if (x == curPlaneNum.x) continue;
					DrawOneVector(Vector3D<size_t>(x, curPlaneNum.y, z),
						vectors(x, curPlaneNum.y, z), cs, minValue, normLimit, normFactor);
				}
			}

			//Перпендикулярно Z
			for (size_t y = 0; y <= vectors.dimY()-1; y++) {
				for (size_t x = 0; x <= vectors.dimX()-1; x++) {
					if (x == curPlaneNum.x || y == curPlaneNum.y) continue;
					DrawOneVector(Vector3D<size_t>(x, y, curPlaneNum.z),
						vectors(x, y, curPlaneNum.z), cs, minValue, normLimit, normFactor);
				}
			}
		
			glColor3f(0.7f, 0.0f, 0.0f);

			glLineStipple(1, 0x00F0);
			glEnable(GL_LINE_STIPPLE);

			glBegin(GL_LINE_LOOP);
			cs.Vertex(curPlaneNum.x, 0, 0);
			cs.Vertex(curPlaneNum.x, vectors.dimY()-1, 0);
			cs.Vertex(curPlaneNum.x, vectors.dimY()-1, vectors.dimZ()-1);
			cs.Vertex(curPlaneNum.x, 0, vectors.dimZ()-1);
			glEnd();

			glBegin(GL_LINE_LOOP);
			cs.Vertex(0, curPlaneNum.y, 0);
			cs.Vertex(vectors.dimX()-1, curPlaneNum.y, 0);
			cs.Vertex(vectors.dimX()-1, curPlaneNum.y, vectors.dimZ()-1);
			cs.Vertex(0, curPlaneNum.y, vectors.dimZ()-1);
			glEnd();

			glBegin(GL_LINE_LOOP);
			cs.Vertex(0, 0, curPlaneNum.z);
			cs.Vertex(vectors.dimX()-1, 0, curPlaneNum.z);
			cs.Vertex(vectors.dimX()-1, vectors.dimY()-1, curPlaneNum.z);
			cs.Vertex(0, vectors.dimY()-1, curPlaneNum.z);
			glEnd();

			glLineStipple(1, 0x0101);
			glBegin(GL_LINES);
			cs.Vertex(0, curPlaneNum.y, curPlaneNum.z);
			cs.Vertex(vectors.dimX()-1, curPlaneNum.y, curPlaneNum.z);

			cs.Vertex(curPlaneNum.x, 0, curPlaneNum.z);
			cs.Vertex(curPlaneNum.x, vectors.dimY()-1, curPlaneNum.z);

			cs.Vertex(curPlaneNum.x, curPlaneNum.y, 0);
			cs.Vertex(curPlaneNum.x, curPlaneNum.y, vectors.dimZ()-1);
			glEnd();

			glDisable(GL_LINE_STIPPLE);

			DrawCostalCude(
				cs.GetGlCoords(Vector3D<size_t>(0, 0, 0)), 
				cs.GetGlCoords(Vector3D<size_t>(vectors.dimX()-1, vectors.dimY()-1, vectors.dimZ()-1)));
		}

		void DrawVectorSpace(const HostData3D<Vector3D<float>>& vectors, float minValue, float maxValue, const Grid3DCoordSys& cs) {
			float normLimit = cs.H();
			float normFactor = normLimit / maxValue;

			glPointSize(2);
			glColor3f(0.0f, 0.0f, 0.0f);
			for (size_t x = 0; x < vectors.dimX(); x++) {
				for (size_t y = 0; y < vectors.dimY(); y++) {
					for (size_t z = 0; z < vectors.dimZ(); z++) {
						DrawOneVector(Vector3D<size_t>(x, y, z), vectors(x, y, z), cs, minValue, normLimit, normFactor);
					}
				}
			}

			glColor3f(0.7f, 0.0f, 0.0f);
			DrawCostalCude(
				cs.GetGlCoords(Vector3D<size_t>(0, 0, 0)),
				cs.GetGlCoords(Vector3D<size_t>(vectors.dimX()-1, vectors.dimY()-1, vectors.dimZ()-1)));
		}

		void DrawImpurityPlain(const HostData2D<float>& impurities, const HostData2D<bool>& isEarths,
													 float minImpurity, float maxImpurity, const Grid3DCoordSys& cs) {
			float *rgb = new float[3];
			glBegin(GL_TRIANGLES);
			for (size_t y = 0; y < impurities.dimY()-1; y++) {
				for (size_t x = 0; x < impurities.dimX()-1; x++) {
					GetColorByImpurity(impurities(x, y), isEarths(x, y), minImpurity, maxImpurity, rgb);
					glColor3fv(rgb);
					cs.Vertex(x, 0, y);

					GetColorByImpurity(impurities(x, y+1), isEarths(x, y+1), minImpurity, maxImpurity, rgb);
					glColor3fv(rgb);
					cs.Vertex(x, 0, y+1);

					GetColorByImpurity(impurities(x+1, y), isEarths(x+1, y), minImpurity, maxImpurity, rgb);
					glColor3fv(rgb);
					cs.Vertex(x+1, 0, y);

					cs.Vertex(x+1, 0, y);

					GetColorByImpurity(impurities(x, y+1), isEarths(x, y+1), minImpurity, maxImpurity, rgb);
					glColor3fv(rgb);
					cs.Vertex(x, 0, y+1);
				
					GetColorByImpurity(impurities(x+1, y+1), isEarths(x+1, y+1), minImpurity, maxImpurity, rgb);
					glColor3fv(rgb);
					cs.Vertex(x+1, 0, y+1);
				}
			}
			glEnd();
		}

		void GetColorByImpurity(float impurity, bool isEarth, float minImpurity, float maxImpurity, float *rgb) {
			if (isEarth) {
				rgb[0] = 0;
				rgb[1] = 255;
				rgb[2] = 0;
				return;
			}

			float impurityPercent = (impurity - minImpurity) / (maxImpurity - minImpurity);
			rgb[0] = 0;
			rgb[1] = 0;
			rgb[2] = 1 - impurityPercent;
		}

		void DrawParticle(float x, float y, float z, float radius) {
			glMatrixMode(GL_MODELVIEW);
			glPushMatrix();
			glTranslatef(x, y, z);
	
			glBegin(GL_LINES);
	
			glVertex3f(0, -radius, 0);
			glVertex3f(0, 0, radius);

			glVertex3f(0, -radius, 0);
			glVertex3f(0, 0, -radius);

			glVertex3f(0, -radius, 0);
			glVertex3f(radius, 0, 0);

			glVertex3f(0, -radius, 0);
			glVertex3f(-radius, 0, 0);

			glVertex3f(0, radius, 0);
			glVertex3f(0, 0, radius);

			glVertex3f(0, radius, 0);
			glVertex3f(0, 0, -radius);

			glVertex3f(0, radius, 0);
			glVertex3f(radius, 0, 0);

			glVertex3f(0, radius, 0);
			glVertex3f(-radius, 0, 0);


			glVertex3f(0, 0, radius);
			glVertex3f(radius, 0, 0);

			glVertex3f(radius, 0, 0);
			glVertex3f(0, 0, -radius);

			glVertex3f(0, 0, -radius);
			glVertex3f(-radius, 0, 0);

			glVertex3f(-radius, 0, 0);
			glVertex3f(0, 0, radius);

			glEnd();

			glPopMatrix();
		}

		void DrawCostalCude(const Vector3D<float>& p0, const Vector3D<float>& p1) {
			glBegin(GL_LINES);

			glVertex3f(p0.x, p0.y, p0.z); glVertex3f(p1.x, p0.y, p0.z);
			glVertex3f(p1.x, p0.y, p0.z); glVertex3f(p1.x, p1.y, p0.z);
			glVertex3f(p1.x, p1.y, p0.z); glVertex3f(p0.x, p1.y, p0.z);
			glVertex3f(p0.x, p1.y, p0.z); glVertex3f(p0.x, p0.y, p0.z);

			glVertex3f(p0.x, p0.y, p1.z); glVertex3f(p1.x, p0.y, p1.z);
			glVertex3f(p1.x, p0.y, p1.z); glVertex3f(p1.x, p1.y, p1.z);
			glVertex3f(p1.x, p1.y, p1.z); glVertex3f(p0.x, p1.y, p1.z);
			glVertex3f(p0.x, p1.y, p1.z); glVertex3f(p0.x, p0.y, p1.z);

			glVertex3f(p0.x, p0.y, p0.z); glVertex3f(p0.x, p0.y, p1.z);
			glVertex3f(p1.x, p0.y, p0.z); glVertex3f(p1.x, p0.y, p1.z);
			glVertex3f(p1.x, p1.y, p0.z); glVertex3f(p1.x, p1.y, p1.z);
			glVertex3f(p0.x, p1.y, p0.z); glVertex3f(p0.x, p1.y, p1.z);

			glEnd();
		}
	}
}