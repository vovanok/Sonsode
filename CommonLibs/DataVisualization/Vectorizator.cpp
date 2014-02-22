#include "Vectorizator.h"

namespace {
	Geometry::Point<float> RealPointFromNode(Geometry::Point<size_t> node, Geometry::Point<float> coordZeroPoint, float h) {
		return Geometry::Point<float>((float)(node.x) * h + coordZeroPoint.x,
																	(float)(node.y) * h + coordZeroPoint.y);
	}

	Geometry::Polygon<float> RealPolygonFromNodes(const Geometry::Polygon<size_t>& nodes,
																								const Geometry::Point<float>& coordLuPoint, float h) {
		Geometry::Polygon<float> result;

		for (auto vertex : nodes.vertexes)
			result.vertexes.push_back(RealPointFromNode(vertex, coordLuPoint, h));

		return result;
	}
}

#pragma region API

std::vector<Geometry::Polygon<float>> Vectorizator::Vectorize(HostData2D<bool> rasterField,
		Geometry::Point<float> coordLuPoint, float h) {

	Vectorizator vtzr(rasterField);
	std::vector<Geometry::Polygon<size_t>> borders = vtzr.GetBorders();
	
	//Преобразование полигонов узлов в поллигоны в реальных координатах
	std::vector<Geometry::Polygon<float>> polygons;
	for (auto border : borders)
		polygons.push_back(RealPolygonFromNodes(border, coordLuPoint, h));

	//Оптимизация полигонов
	for(auto& polygon : polygons)
		if (polygon.vertexes.size() > 3)		
			polygon.Optimize(h / 2.0f);

	return polygons;
}

void Vectorizator::DiscretizeAndAppend(Geometry::Polygon<float> bound, Geometry::Point<float> luPoint,
																			 float h, bool boundIsOuter, HostData2D<bool>& field) {
	Geometry::Point<size_t> node(0, 0);
	for (node.x = 0; node.x < field.dimX(); node.x++) {
		for (node.y = 0; node.y < field.dimY(); node.y++) {

			if (bound.PointInSide(RealPointFromNode(node, luPoint, h)))
				field(node.x, node.y) = boundIsOuter;
		}
	}
}

#pragma endregion

#pragma region Vectorizator implementation

Vectorizator::Vectorizator(HostData2D<bool> rasterField) {
	rasterWorkField = HostData2D<RasterItem>(rasterField.dimX(), rasterField.dimY());

	for (size_t x = 0; x <= rasterField.dimX() - 1; x++)
		for (size_t y = 0; y <= rasterField.dimY() - 1; y++)
			rasterWorkField(x, y) = RasterItem(rasterField(x, y));
}

Vectorizator::~Vectorizator() {
	rasterWorkField.Erase();
}

std::vector<Geometry::Polygon<size_t>> Vectorizator::GetBorders() {
	std::vector<Geometry::Polygon<size_t>> borders;
		
	//Поиск очередной точки, являющейся границей пожара
	Geometry::Point<size_t> nextStartP = GetFirstBorderNotCrossNode();

	//Пока есть граничные точки, не включенные в полигоны
	while (!nextStartP.isEmpty) {
		Geometry::Polygon<size_t> nextPolygon = TracePolygonFromStartNode(nextStartP);
		if (nextPolygon.vertexes.size() >= 3)
			borders.push_back(nextPolygon);

		nextStartP = GetFirstBorderNotCrossNode();
	}

	return borders;
}

Geometry::Point<size_t> Vectorizator::GetFirstBorderNotCrossNode() {
	Geometry::Point<size_t> node(0, 0);
	for (node.y = 0; node.y < rasterWorkField.dimY(); node.y++) {
		for (node.x = 0; node.x < rasterWorkField.dimX(); node.x++) {
			if (IsBoundNode(node) && !IsCross(node))
			return node;
		}
	}

	return Geometry::Point<size_t>::Empty();
}

bool Vectorizator::IsBoundNode(Geometry::Point<size_t> point) {
	if (!IsValue(point))
		return false;

	if (point.x == 0 || point.x == rasterWorkField.dimX() - 1 ||
			point.y == 0 || point.y == rasterWorkField.dimY() - 1)
		return true;

	if (!IsValue(point.Moved(-1, 0)) || 	!IsValue(point.Moved(1, 0)) ||
			!IsValue(point.Moved(0, -1)) || 	!IsValue(point.Moved(0, 1)))
		return true;

	return false;
}

bool Vectorizator::IsValue(Geometry::Point<size_t> point) {
	return rasterWorkField(point.x, point.y).isValue;
}
        
bool Vectorizator::IsCross(Geometry::Point<size_t> point) {
	return rasterWorkField(point.x, point.y).isCross;
}

void Vectorizator::CrossPoint(Geometry::Point<size_t> point) {
	rasterWorkField(point.x, point.y).isCross = true;
}

Geometry::Polygon<size_t> Vectorizator::TracePolygonFromStartNode(Geometry::Point<size_t> startNode) {
	Geometry::Polygon<size_t> resultPolygon;

	if (startNode.isEmpty || !IsBoundNode(startNode))
		return resultPolygon;

	resultPolygon = TraceNodesFromStartNode(startNode, BorderType::Right);
	if (resultPolygon.vertexes.size() <= 1)
		resultPolygon = TraceNodesFromStartNode(startNode, BorderType::Left);

	return resultPolygon;
}

Geometry::Polygon<size_t> Vectorizator::TraceNodesFromStartNode(Geometry::Point<size_t> startNode, BorderType borderType) {
	Geometry::Polygon<size_t> resultNodes;

	if (startNode.isEmpty)
		return resultNodes;

	Geometry::Point<size_t> node(startNode);

	//Цикл трассировки от стартовой точки
	do {
		resultNodes.vertexes.push_back(node);
		CrossPoint(node); //Проходим узел

		node = GetNextBorderNodeFromCurrent(node, borderType);
	} while (!(node.Equal(startNode))	&& !node.isEmpty);
	//Цикл трассировки продолжается:
	//	- пока не пришли обратно в стартовую точку;
	//	- пока есть следующая точка для трассировки
		
	return resultNodes;
}

Geometry::Point<size_t> Vectorizator::GetNextBorderNodeFromCurrent(Geometry::Point<size_t> curNode, BorderType borderType) {
	Geometry::Point<size_t> lInd = curNode.Moved(-1, 0);
	Geometry::Point<size_t> ulInd = curNode.Moved(-1, -1);
	Geometry::Point<size_t> uInd = curNode.Moved(0, -1);
	Geometry::Point<size_t> urInd = curNode.Moved(1, -1);
	Geometry::Point<size_t> rInd = curNode.Moved(1, 0);
	Geometry::Point<size_t> drInd = curNode.Moved(1, 1);
	Geometry::Point<size_t> dInd = curNode.Moved(0, 1);
	Geometry::Point<size_t> dlInd = curNode.Moved(-1, 1);

	//Левая точка
	if (!IsCross(lInd) && IsBoundNode(lInd) &&
			((borderType == BorderType::Right && (!IsValue(dlInd) || !IsValue(dInd))) ||
			 (borderType == BorderType::Left && (!IsValue(ulInd) || !IsValue(uInd)))))
		return lInd;

	//Верхняя левая точка
	if (!IsCross(ulInd) && IsBoundNode(ulInd) &&
			((borderType == BorderType::Right && !IsValue(lInd)) ||
			 (borderType == BorderType::Left && !IsValue(uInd))))
		return ulInd;

	//Верняя точка
	if (!IsCross(uInd) && IsBoundNode(uInd) &&
			((borderType == BorderType::Right && (!IsValue(lInd) || !IsValue(ulInd))) ||
			 (borderType == BorderType::Left && (!IsValue(rInd) || !IsValue(urInd)))))
		return uInd;
		
	//Верхняя правая точка
	if (!IsCross(urInd) && IsBoundNode(urInd) &&
			((borderType == BorderType::Right && !IsValue(uInd)) ||
			 (borderType == BorderType::Left && !IsValue(rInd))))
		return urInd;

	//Правая точка
	if (!IsCross(rInd) && IsBoundNode(rInd) &&
			((borderType == BorderType::Right && (!IsValue(uInd) || !IsValue(urInd))) ||
			 (borderType == BorderType::Left && (!IsValue(drInd) || !IsValue(dInd)))))
		return rInd;

	//Нижняя правая точка
	if (!IsCross(drInd) && IsBoundNode(drInd) &&
			((borderType == BorderType::Right && !IsValue(rInd)) ||
			 (borderType == BorderType::Left && !IsValue(dInd))))
		return drInd;

	//Нижняя точка
	if (!IsCross(dInd) && IsBoundNode(dInd) &&
			((borderType == BorderType::Right && (!IsValue(rInd) || !IsValue(drInd))) ||
			 (borderType == BorderType::Left && (!IsValue(lInd) || !IsValue(dlInd)))))
		return dInd;

	//Левая нижняя точка
	if (!IsCross(dlInd) && IsBoundNode(dlInd) &&
			((borderType == BorderType::Right && !IsValue(dInd)) ||
			 (borderType == BorderType::Left && !IsValue(lInd))))
		return dlInd;

	//Не нашлось соседней граничной точки
	return Geometry::Point<size_t>::Empty();
}

#pragma endregion