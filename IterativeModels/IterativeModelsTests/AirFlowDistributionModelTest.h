#pragma once

#include "AirFlowModel.h"
#include "GraphicUtils.h"

using Sonsode::HostData3D;
using namespace AirFlow;

enum class VectorViewType {
	SPACE = 0,
	DISSECTED_SPACE = 1
};

enum class ViewMode {
	NONE = 0,
	RO_SPACE = 1,
	T_SPACE = 2
};

class AirFlowDistributionModelTest : public AirFlowModel, public IPresentable {
private:
	VectorViewType vectorsView;
	ViewMode viewMode;
	ColorPalette roCp;
	ColorPalette tCp;

	float minVectorValue, maxVectorValue;

	HostData3D<Vector3D<float>> vectors;
	HostData3D<float> buffer;

	Grid3DCoordSys cs;
public:
	Vector3D<size_t> curPlane;
	
	AirFlowDistributionModelTest(AirFlowConsts consts, AirFlowDataH data)
			: AirFlowModel(consts, data),
				roCp(ColorPalette(Color(0.0f, 0.0f, 0.0f, 1.0f), Color(1.0f, 1.0f, 1.0f, 1.0f), 1.2f, 1.3f)),
				tCp(ColorPalette(Color(0.0f, 0.0f, 1.0f, 1.0f), Color(1.0f, 0.0f, 0.0f, 1.0f), 0.0f, 500.0f)),
				cs(Grid3DCoordSys(Vector3D<size_t>(_data.dimX(), _data.dimY(), _data.dimZ()), 35.0f, _consts.H)),
				vectors(HostData3D<Vector3D<float>>(data.dimX(), data.dimY(), data.dimZ())),
				buffer(HostData3D<float>(data.dimX(), data.dimY(), data.dimZ())) {

		tCp.AddUpdControlColor(25, Color(0.0f, 1.0f, 1.0f, 1.0f));
		tCp.AddUpdControlColor(50, Color(0.0f, 1.0f, 0.0f, 1.0f));
		tCp.AddUpdControlColor(75, Color(1.0f, 1.0f, 0.0f, 1.0f));

		minVectorValue = 1.0f;
		maxVectorValue = 20.0f;

		vectorsView = VectorViewType::DISSECTED_SPACE;
		viewMode = ViewMode::NONE;
	}

	~AirFlowDistributionModelTest() {
		vectors.Erase();
		buffer.Erase();
	}

	void SetVectorsView(VectorViewType viewType) {
		vectorsView = viewType;
	};

	void ChangeVectorsView() {
		switch (vectorsView) {
		case VectorViewType::SPACE:
			SetVectorsView(VectorViewType::DISSECTED_SPACE); break;
		case VectorViewType::DISSECTED_SPACE:
			SetVectorsView(VectorViewType::SPACE); break;
		default:
			SetVectorsView(VectorViewType::DISSECTED_SPACE);
		}
	}

	void SetViewMode(ViewMode viewMode) {
		this->viewMode = viewMode;
	}

	void ChangeViewMode() {
		switch(viewMode) {
		case ViewMode::NONE:
			SetViewMode(ViewMode::RO_SPACE); break;
		case ViewMode::RO_SPACE:
			SetViewMode(ViewMode::T_SPACE); break;
		case ViewMode::T_SPACE:
			SetViewMode(ViewMode::NONE); break;
		default:
			SetViewMode(ViewMode::NONE); break;
		}
	}

	virtual void Draw() {
		SynchronizeWithGpu();

		for (size_t z = 0; z < _data.dimZ(); z++)
			for (size_t y = 0; y < _data.dimY(); y++)
				for (size_t x = 0; x < _data.dimX(); x++)
					vectors(x, y, z) = Vector3D<float>(_data.ux(x, y, z), _data.uy(x, y, z), _data.uz(x, y, z));

		switch(vectorsView) {
		case VectorViewType::DISSECTED_SPACE:
			DataVisualization::Graphic::DrawDissectedVectorSpace(vectors, minVectorValue, maxVectorValue, curPlane, cs); break;
		case VectorViewType::SPACE:
			DataVisualization::Graphic::DrawVectorSpace(vectors, minVectorValue, maxVectorValue, cs); break;
		}

		for (size_t z = 0; z < _data.dimZ(); z++)
			for (size_t y = 0; y < _data.dimY(); y++)
				for (size_t x = 0; x < _data.dimX(); x++)
					buffer(x, y, z) = (viewMode == ViewMode::RO_SPACE) ? _data.t(x, y, z) : _data.ro(x, y, z);
		DataVisualization::Graphic::DrawColoredSpace(buffer, curPlane, (viewMode == ViewMode::RO_SPACE) ? tCp : roCp, cs);
	}

	virtual void Impact(char keyCode, int button, int state, int x, int y) {
		switch (keyCode) {
			//X+ 'J'
			case 106: case 238: case 74: case 206:
				if (curPlane.x < _data.dimX() - 1) curPlane.x++; break;
			//X- 'G'
			case 103: case 71: case 239: case 207:
				if (curPlane.x > 0) curPlane.x--; break;
			//Y+ 'H'
			case 104: case 240: case 72: case 208:
				if (curPlane.y < _data.dimY() - 1) curPlane.y++; break;
			//Y- 'Y'
			case 237: case 121: case 89: case 205:
				if (curPlane.y > 0) curPlane.y--; break;
			//Z+ 'I'
			case 107: case 75: case 235: case 203:
				if (curPlane.z < _data.dimZ() - 1) curPlane.z++; break;
			//Z- 'K'
			case 105: case 248: case 73: case 216:
				if (curPlane.z > 0) curPlane.z--; break;
			//Change vectors view 'U'
			case 117: case 85: case 227: case 195:
				ChangeVectorsView(); break;
			//Change view mode 'T'
			case 'T': case 't': case 'Å': case 'å':
				ChangeViewMode(); break;
		}
	}
};