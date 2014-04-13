#pragma once

#include "ColorPalette.h"
#include "HostData.hpp"
#include "Vectors.hpp"
#include "Grid3DCoordSys.h"

namespace DataVisualization {
	namespace Graphic {
		using Sonsode::HostData2D;
		using Sonsode::HostData3D;
		using Sonsode::Vector3D;
		using DataVisualization::Geometry::Grid3DCoordSys;
		using DataVisualization::Graphic::Color;

		void DrawImpurityPlain(const HostData2D<float>& impurities, const HostData2D<bool>& isEarths,
													 float minImpurity, float maxImpurity, const Grid3DCoordSys& cs);
		void DrawParticle(float x, float y, float z, float radius);
		void GetColorByImpurity(float impurity, bool isEarth, float minImpurity, float maxImpurity, float* rgb);
		void DrawCostalCude(const Vector3D<float>& p0, const Vector3D<float>& p1);
		void DrawVectorSpace(const HostData3D<Vector3D<float>>& vectors, float minValue, float maxValue, const Grid3DCoordSys& cs);
		void DrawDissectedVectorSpace(const HostData3D<Vector3D<float>>& vectors, float minValue, float maxValue,
																	const Vector3D<size_t>& curPlaneNum, const Grid3DCoordSys& cs);
		void DrawColoredSpace(const HostData3D<float>& values, const Vector3D<size_t>& curPlaneNum,
													const ColorPalette& cp, const Grid3DCoordSys& cs);
	}
}