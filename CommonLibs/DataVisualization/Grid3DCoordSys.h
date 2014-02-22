#ifndef GRID3DCOORDSYS_HPP
#define GRID3DCOORDSYS_HPP

#include <gl/glut.h>
#include "Vectors.hpp"

class Grid3DCoordSys {
private:
	Vector3D<float> gl0Coords;
	Vector3D<float> glSize;
	float cellSize;
	float h;
	void Initialization(Vector3D<size_t> dim, float maxGlSize, float h);
public:
	Grid3DCoordSys(Vector3D<size_t> dim);
	Grid3DCoordSys(Vector3D<size_t> dim, float maxGlSize);
	Grid3DCoordSys(Vector3D<size_t> dim, float maxGlSize, float h);
	Vector3D<float> GetGrid3DCoords(Vector3D<size_t> cellInds) const;
	Vector3D<float> GetGlCoords(Vector3D<size_t> cellInds) const;
	Vector3D<float> GetGlCoords(Vector3D<float> srcCoords) const;
	void Vertex(size_t x, size_t y, size_t z) const;
	void Vertex(float x, float y, float z) const;
	void Vertex(Vector3D<size_t> p) const;
	void Vertex(Vector3D<float> p) const;
	float GetCellGlSize() const;
	float H() const;
};

#endif