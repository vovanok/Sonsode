#include "Vectorizator.h"

namespace {
	Point<float> RealPointFromNode(Point<size_t> node, Point<float> coordZeroPoint, float h) {
		return Point<float>((float)(node.x) * h + coordZeroPoint.x,
												(float)(node.y) * h + coordZeroPoint.y);
	}

	Polygon<float> RealPolygonFromNodes(const Polygon<size_t>& nodes, const Point<float>& coordLuPoint, float h) {
		Polygon<float> result;

		for (auto vertex : nodes.vertexes)
			result.vertexes.push_back(RealPointFromNode(vertex, coordLuPoint, h));

		return result;
	}
}

namespace DataVisualization {
	namespace Geometry {
		#pragma region API

		vector<Polygon<float>> Vectorizator::Vectorize(HostData2D<bool> rasterField, Point<float> coordLuPoint, float h) {
			Vectorizator vtzr(rasterField);
			vector<Polygon<size_t>> borders = vtzr.GetBorders();
	
			//Преобразование полигонов узлов в поллигоны в реальных координатах
			vector<Polygon<float>> polygons;
			for (auto border : borders)
				polygons.push_back(RealPolygonFromNodes(border, coordLuPoint, h));

			//Оптимизация полигонов
			for (auto& polygon : polygons) {
				if (polygon.vertexes.size() > 3)		
					polygon.Optimize(h / 2.0f);
			}

			return polygons;
		}

		void Vectorizator::DiscretizeAndAppend(Polygon<float> bound, Point<float> luPoint, float h, bool boundIsOuter, HostData2D<bool>& field) {
			Point<size_t> node(0, 0);
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

			for (size_t x = 0; x <= rasterField.dimX() - 1; x++) {
				for (size_t y = 0; y <= rasterField.dimY() - 1; y++) {
					rasterWorkField(x, y) = RasterItem(rasterField(x, y));
				}
			}
		}

		Vectorizator::~Vectorizator() {
			rasterWorkField.Erase();
		}

		vector<Polygon<size_t>> Vectorizator::GetBorders() {
			vector<Polygon<size_t>> borders;
		
			//Поиск очередной точки, являющейся границей пожара
			Point<size_t> nextStartP = GetFirstBorderNotCrossNode();

			//Пока есть граничные точки, не включенные в полигоны
			while (!nextStartP.isEmpty) {
				Polygon<size_t> nextPolygon = TracePolygonFromStartNode(nextStartP);
				if (nextPolygon.vertexes.size() >= 3)
					borders.push_back(nextPolygon);

				nextStartP = GetFirstBorderNotCrossNode();
			}

			return borders;
		}

		Point<size_t> Vectorizator::GetFirstBorderNotCrossNode() {
			Point<size_t> node(0, 0);
			for (node.y = 0; node.y < rasterWorkField.dimY(); node.y++) {
				for (node.x = 0; node.x < rasterWorkField.dimX(); node.x++) {
					if (IsBoundNode(node) && !IsCross(node))
					return node;
				}
			}

			return Point<size_t>::Empty();
		}

		bool Vectorizator::IsBoundNode(Point<size_t> point) {
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

		bool Vectorizator::IsValue(Point<size_t> point) {
			return rasterWorkField(point.x, point.y).isValue;
		}
        
		bool Vectorizator::IsCross(Point<size_t> point) {
			return rasterWorkField(point.x, point.y).isCross;
		}

		void Vectorizator::CrossPoint(Point<size_t> point) {
			rasterWorkField(point.x, point.y).isCross = true;
		}

		Polygon<size_t> Vectorizator::TracePolygonFromStartNode(Point<size_t> startNode) {
			Polygon<size_t> resultPolygon;

			if (startNode.isEmpty || !IsBoundNode(startNode))
				return resultPolygon;

			resultPolygon = TraceNodesFromStartNode(startNode, BorderType::Right);
			if (resultPolygon.vertexes.size() <= 1)
				resultPolygon = TraceNodesFromStartNode(startNode, BorderType::Left);

			return resultPolygon;
		}

		Polygon<size_t> Vectorizator::TraceNodesFromStartNode(Point<size_t> startNode, BorderType borderType) {
			Polygon<size_t> resultNodes;

			if (startNode.isEmpty)
				return resultNodes;

			Point<size_t> node(startNode);

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

		Point<size_t> Vectorizator::GetNextBorderNodeFromCurrent(Point<size_t> curNode, BorderType borderType) {
			Point<size_t> lInd = curNode.Moved(-1, 0);
			Point<size_t> ulInd = curNode.Moved(-1, -1);
			Point<size_t> uInd = curNode.Moved(0, -1);
			Point<size_t> urInd = curNode.Moved(1, -1);
			Point<size_t> rInd = curNode.Moved(1, 0);
			Point<size_t> drInd = curNode.Moved(1, 1);
			Point<size_t> dInd = curNode.Moved(0, 1);
			Point<size_t> dlInd = curNode.Moved(-1, 1);

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
			return Point<size_t>::Empty();
		}

		#pragma endregion
	}
}