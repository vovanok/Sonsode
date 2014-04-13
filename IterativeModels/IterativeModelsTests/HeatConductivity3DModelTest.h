#pragma once

#include "Heat3DModel.h"
#include "IPresentable.h"
#include "GraphicUtils.h"

using namespace Heat3D;

class HeatConductivity3DModelTest : public Heat3DModel, public IPresentable {
public:
	Vector3D<size_t> curPlane;

	HeatConductivity3DModelTest(HostData3D<float> t, float h, float a, float tau)
			: Heat3DModel(t, h, a, tau),
				cs(Grid3DCoordSys(Vector3D<size_t>(t.dimX(), t.dimY(), t.dimZ()))),
				cp(ColorPalette(Color(0.0f, 0.0f, 1.0f, 1.0f), Color(1.0f, 0.0f, 0.0f, 1.0f), 0.0f, 500.0f)) {

		cp.AddUpdControlColor(25, Color(0.0f, 1.0f, 1.0f, 1.0f));
		cp.AddUpdControlColor(50, Color(0.0f, 1.0f, 0.0f, 1.0f));
		cp.AddUpdControlColor(75, Color(1.0f, 1.0f, 0.0f, 1.0f));
	}

	virtual void Draw() {
		SynchronizeWithGpu();
		DataVisualization::Graphic::DrawColoredSpace(t(), curPlane, cp, cs);
	}

	virtual void Impact(char keyCode, int button, int state, int x, int y) {
		switch (keyCode) {
			case 106: case 238: case 74: case 206: //X++
				if (curPlane.x < t().dimX() - 1) curPlane.x++;
				break;
			case 103: case 71: case 239: case 207: //X--
				if (curPlane.x > 0) curPlane.x--;
				break;
			case 104: case 240: case 72: case 208: //Y++
				if (curPlane.y < t().dimY() - 1) curPlane.y++;
				break;
			case 237: case 121: case 89: case 205: //Y--
				if (curPlane.y > 0) curPlane.y--;
				break;
			case 107: case 75: case 235: case 203: //Z++
				if (curPlane.z < t().dimZ() - 1) curPlane.z++;
				break;
			case 105: case 248: case 73: case 216: //Z--
				if (curPlane.z > 0) curPlane.z--;
				break;
		}
	}
private:
	ColorPalette cp;
	Grid3DCoordSys cs;
};