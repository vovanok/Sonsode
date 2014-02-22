#pragma once

#include "FireSpreadSimpleModel.h"
#include "GraphicUtils.h"
#include "IPresentable.h"

class FireSpreadSimpleModelTest : public FireSpreadSimpleModel, public IPresentable {
public:
	FireSpreadSimpleModelTest(FireSpreadConsts consts, FireSpreadDataH data)
		: FireSpreadSimpleModel(consts, data),
			cs(Grid3DCoordSys(Vector3D<size_t>(data.dimX(), 1, data.dimY()))),
			cp(ColorPalette(Color(0.0f, 0.0f, 1.0f, 1.0f), Color(1.0f, 0.0f, 0.0f, 1.0f), 0.0f, 200.0f)) {

		cp.AddUpdControlColor(25, Color(0.0f, 1.0f, 1.0f, 1.0f));
		cp.AddUpdControlColor(50, Color(0.0f, 1.0f, 0.0f, 1.0f));
		cp.AddUpdControlColor(75, Color(1.0f, 1.0f, 0.0f, 1.0f));
	}

	virtual void Draw() {
		SynchronizeWithGpu();
		
		Sonsode::HostData3D<float> drawingData(_data.dimX(), 1, _data.dimY());
		for (size_t x = 0; x <= _data.dimX() - 1; x++)
			for (size_t y = 0; y <= _data.dimY() - 1; y++)
				drawingData(x, 0, y) = _data.t(x, y);

		GraphicUtils::DrawColoredSpace(drawingData, Vector3D<size_t>(0, 0, 0), cp, cs);
	}

	virtual void Impact(char keyCode, int button, int state, int x, int y) { }

private:
	ColorPalette cp;
	Grid3DCoordSys cs;
};