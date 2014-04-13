#include "Grid3DCoordSys.h"

namespace DataVisualization {
	namespace Geometry {
		void Grid3DCoordSys::Initialization(Vector3D<size_t> dim, float maxGlSize, float h) {
			cellSize = maxGlSize / (float)std::max(std::max(dim.x, dim.y), dim.z);
			glSize = Vector3D<float>((float)dim.x * cellSize, (float)dim.z * cellSize, (float)dim.y * cellSize);
			gl0Coords = Vector3D<float>(-glSize.x/2, glSize.y/2, -glSize.z/2);
			this->h = (h < 0.0f) ? cellSize : h;
		}

		Grid3DCoordSys::Grid3DCoordSys(Vector3D<size_t> dim) {
			Initialization(dim, 30, -1.0f);
		}

		Grid3DCoordSys::Grid3DCoordSys(Vector3D<size_t> dim, float maxGlSize) {
			Initialization(dim, maxGlSize, -1.0f);
		}

		Grid3DCoordSys::Grid3DCoordSys(Vector3D<size_t> dim, float maxGlSize, float h) {
			Initialization(dim, maxGlSize, h);
		}

		Vector3D<float> Grid3DCoordSys::GetGrid3DCoords(Vector3D<size_t> cellInds) const {
			return Vector3D<float>(cellInds.x * h, cellInds.y * h, cellInds.z * h);
		}

		Vector3D<float> Grid3DCoordSys::GetGlCoords(Vector3D<size_t> cellInds) const {
			return GetGlCoords(GetGrid3DCoords(cellInds));
		}

		Vector3D<float> Grid3DCoordSys::GetGlCoords(Vector3D<float> srcCoords) const {
			float factor = cellSize / h;
			return Vector3D<float>(			
				factor * srcCoords.x + gl0Coords.x,
				-(factor * srcCoords.z) + gl0Coords.y,
				factor * srcCoords.y + gl0Coords.z);
		}

		void Grid3DCoordSys::Vertex(size_t x, size_t y, size_t z) const {
			Vertex(Vector3D<size_t>(x, y, z));
		}
	
		void Grid3DCoordSys::Vertex(float x, float y, float z) const {
			Vertex(Vector3D<float>(x, y, z));
		}

		void Grid3DCoordSys::Vertex(Vector3D<size_t> p) const {
			Vector3D<float> glCoords = GetGlCoords(p);
			glVertex3f(glCoords.x, glCoords.y, glCoords.z);
		}
	
		void Grid3DCoordSys::Vertex(Vector3D<float> p) const {
			Vector3D<float> glCoords = GetGlCoords(p);
			glVertex3f(glCoords.x, glCoords.y, glCoords.z);
		}

		float Grid3DCoordSys::GetCellGlSize() const {
			return cellSize;
		}

		float Grid3DCoordSys::H() const {
			return h;
		}
	}
}