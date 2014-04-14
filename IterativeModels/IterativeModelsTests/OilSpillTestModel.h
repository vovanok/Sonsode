#pragma once

#include "OilSpillModel.h"
#include "GraphicUtils.h"
#include "IPresentable.h"

using Sonsode::HostData2D;
using namespace OilSpill;

class OilSpillTestModel : public OilSpillModel, public IPresentable {
public:
	OilSpillTestModel(OilSpillConsts consts, OilSpillDataH data)
		: OilSpillModel(consts, data),
			cs(Grid3DCoordSys(Vector3D<size_t>(data.dimX(), 1, data.dimY()))) { }

	virtual ~OilSpillTestModel() { }

	virtual void Draw() {
		try {
			SynchronizeWithGpu();

			HostData2D<float> impurities(dimX(), dimY());
			HostData2D<bool> isEarths(dimX(), dimY());
		
			for (size_t x = 0; x < dimX(); x++) {
				for (size_t y = 0; y < dimY(); y++) {
					impurities(x, y) = impurity(x, y);
					isEarths(x, y) = (deep(x, y) <= 0);
				}
			}
		
			DataVisualization::Graphic::DrawImpurityPlain(impurities, isEarths, 0.0f, 100.0f, cs);

			impurities.Erase();
			isEarths.Erase();
		} catch(std::exception e) {
			std::cout << e.what() << std::endl;
		}
	}

	virtual void Impact(char keyCode, int button, int state, int x, int y) { }

protected:
	Grid3DCoordSys cs;
};