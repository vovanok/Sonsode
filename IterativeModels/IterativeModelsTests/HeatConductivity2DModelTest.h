#pragma once

#include "Heat2DModel.h"
#include "IPresentable.h"
#include "GraphicUtils.h"

using namespace Heat2D;

class HeatConductivity2DModelTest : public Heat2DModel, public IPresentable {
public:
	HeatConductivity2DModelTest(HostData2D<float> t, float h, float a, float tau)
			: Heat2DModel(t, h, a, tau),
				cs(Grid3DCoordSys(Vector3D<size_t>(t.dimX(), 1, t.dimY()))),
				cp(ColorPalette(Color(0.0f, 0.0f, 1.0f, 1.0f), Color(1.0f, 0.0f, 0.0f, 1.0f), 0.0f, 200.0f)) {
		
		cp.AddUpdControlColor(25, Color(0.0f, 1.0f, 1.0f, 1.0f));
		cp.AddUpdControlColor(50, Color(0.0f, 1.0f, 0.0f, 1.0f));
		cp.AddUpdControlColor(75, Color(1.0f, 1.0f, 0.0f, 1.0f));
	}

	virtual void Draw() {
		SynchronizeWithGpu();
		Sonsode::HostData3D<float> drawingData(t().dimX(), 1, t().dimY());

		for (size_t x = 0; x < t().dimX(); x++)
			for (size_t y = 0; y < t().dimY(); y++)
				drawingData(x, 0, y) = t()(x, y);
		
		DataVisualization::Graphic::DrawColoredSpace(drawingData, Vector3D<size_t>(0, 0, 0), cp, cs);
	}

	virtual void Impact(char keyCode, int button, int state, int x, int y) { }

private:
	ColorPalette cp;
	Grid3DCoordSys cs;
};
