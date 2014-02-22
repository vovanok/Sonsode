#pragma once

#include "OilSpillageImprovedModel.h"
#include "GraphicUtils.h"
#include "IPresentable.h"

using Sonsode::HostData2D;

class OilSpillageImprovedModelTest : public OilSpillageImprovedModel, public IPresentable {
public:
	OilSpillageImprovedModelTest(OilSpillageConsts consts, OilSpillageDataH data)
		: OilSpillageImprovedModel(consts, data),
			cs(Grid3DCoordSys(Vector3D<size_t>(data.dimX(), 1, data.dimY()))) { }

	virtual ~OilSpillageImprovedModelTest() { }

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
		
			GraphicUtils::DrawImpurityPlain(impurities, isEarths, 0.0f, 100.0f, cs);

			impurities.Erase();
			isEarths.Erase();
		} catch(std::string e) {
			std::cout << e << std::endl;
		}
	}

	virtual void Impact(char keyCode, int button, int state, int x, int y) { }

protected:
	Grid3DCoordSys cs;
};