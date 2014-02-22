#pragma once

#include <math.h>
#include <vector>
#include <algorithm>

namespace Geometry {
	template<class T> class Rect;

	template<class T>
	class Point {
	public:
		T x;
		T y;
		bool isEmpty;

		static Point<T> Empty() { return Point(true); }

		Point() : x(0), y(0), isEmpty(false) { }
		Point(bool isEmpty) : x(0), y(0), isEmpty(isEmpty) { }
		Point(T x, T y) : x(x), y(y), isEmpty(false) { }
		Point(const Point<T>& point) : x(point.x), y(point.y), isEmpty(false) { }

		bool Equal(const Point<T> &comparePoint) const;
		Point<T> Moved(const T dx, const T dy) const;
		float DistanceToLine(const float& a, const float& b, const float& c) const;
		float DistanceToLine(const Point<T>& linePoint1, const Point<T>& linePoint2) const;
		float DistanceToPoint(const Point<T>& point) const;
		void NormalizeCoordinate(const Rect<T>& src, const Rect<T>& dst);
	};

	template<class T>
	class Rect {
	public:
		Point<T> point1;
		Point<T> point2;

		Rect() : point1(Point<T>(0, 0)), point2(Point<T>(0, 0)) { }
		Rect(Point<T> point1, Point<T> point2) : point1(point1), point2(point2) { }
	};

	template<class T>
	class Polygon {
	public:
		std::vector<Point<T>> vertexes;

		Polygon() : vertexes(std::vector<Point<T>>(0)) { }
		Polygon(const std::vector<Point<T>>& vertexes) : vertexes(vertexes) { }
		~Polygon();
		bool PointInSide(const Point<T>& point) const;
		bool PointInSide(T x, T y) const;
		Rect<T> GetClearanceBorders() const;
		void Optimize(float maxEps);
		void Normalize(const Rect<T>& src, const Rect<T>& dst);

	private:
		bool IsFormOneIntersect(size_t startVertNum, size_t endVertNum, float maxEps) const;
		size_t GetNextOptimizeVertNumFromCurrent(size_t currentNum, float maxEps) const;
	};

	//bool IsIntersect(Polygon otherPolygon)
	//{
	//	int countVertexes = GetCountVetexes();
	//	for(int curVertNum = 0; curVertNum < countVertexes; curVertNum++)
	//	{
	//		if (otherPolygon.PointInSide(GetVertex(curVertNum)))
	//			return true;
	//	}
	//	return false;
	//}
	//
	//void Union(Polygon otherPolygon)
	//{
	//	if (!IsIntersect(otherPolygon))
	//		return;
	//	
	//	std::vector<Point<float>> newMyPoints;
	//	for (int curMyPointNum = 0; curMyPointNum < countVerts; curMyPointNum++)
	//	{
	//		if (!otherPolygon.PointInSide(GetVertex(curMyPointNum)))
	//			newMyPoints.push_back(GetVertex(curMyPointNum));
	//	}
	//	verts = newMyPoints.data();
	//	countVerts = newMyPoints.size();
	//}

	#pragma region Implementation

	#pragma region Point
	template<class T>
	bool Point<T>::Equal(const Point<T> &comparePoint) const {
		return (isEmpty && comparePoint.isEmpty) ||
			(!isEmpty && !comparePoint.isEmpty && (x == comparePoint.x && y == comparePoint.y));
	}

	template<class T>
	Point<T> Point<T>::Moved(const T dx, const T dy) const {
		return Point<T>(x + dx, y + dy);
	}

	template<class T>
	float Point<T>::DistanceToLine(const float& a, const float& b, const float& c) const {
		return abs(a * x + b * y + c) / sqrtf(powf(a, 2.0f) + powf(b, 2.0f));
	}

	template<class T>
	float Point<T>::DistanceToLine(const Point<T>& linePoint1, const Point<T>& linePoint2) const {
		float a = linePoint2.y - linePoint1.y;
		float b = linePoint1.x - linePoint2.x;
		float c = -linePoint1.x * (linePoint2.y - linePoint1.y)
			+ linePoint1.y * (linePoint2.x - linePoint1.x);

		return DistanceToLine(a, b, c);
	}

	template<class T>
	float Point<T>::DistanceToPoint(const Point<T>& point) const {
		return sqrtf(powf(x - point.x) + powf(y - point.y));
	}

	template<class T>
	void Point<T>::NormalizeCoordinate(const Rect<T>& src, const Rect<T>& dst) {
		x = dst.point1.x + (x - src.point1.x) *
			((dst.point2.x - dst.point1.x) / (src.point2.x - src.point1.x));
		y = dst.point1.y + (y - src.point1.y) *
			((dst.point2.y - dst.point1.y) / (src.point2.y - src.point1.y));
	}
	#pragma endregion

	#pragma region Polygon
	template<class T>
	Polygon<T>::~Polygon() {
		vertexes.clear();
	}

	template<class T>
	bool Polygon<T>::PointInSide(const Point<T>& point) const {
		return PointInSide(point.x, point.y);
	}

	template<class T>
	bool Polygon<T>::PointInSide(T x, T y) const {
		int countVerts = vertexes.size();
		int prevP_ind = countVerts - 1;
		bool result = false;

		for (int nextP_ind = 0; nextP_ind < countVerts; nextP_ind++) {
			Point<T> prevP = vertexes[prevP_ind];
			Point<T> nextP = vertexes[nextP_ind];

			if (((nextP.y < y && y <= prevP.y) || (prevP.y < y && y <= nextP.y)) &&
					(nextP.x + (y - nextP.y) / (prevP.y - nextP.y) * 	(prevP.x - nextP.x) < x)) {
				result = !result;
			}
			prevP_ind = nextP_ind;
		}
		return result;
	}

	template<class T>
	Rect<T> Polygon<T>::GetClearanceBorders() const {
		if (vertexes.size() == 0)
			return Rect<T>(Point<T>(0, 0), Point<T>(0, 0));

		Rect<T> result(vertexes[0], vertexes[0]);

		for (Point<T> vertex : vertexes) {
			result.point1.x = std::min(result.point1.x, vertex.x);
			result.point1.y = std::min(result.point1.y, vertex.y);
			result.point2.x = std::max(result.point2.x, vertex.x);
			result.point2.y = std::max(result.point2.y, vertex.y);
		}

		return result;
	}

	template<class T>
	void Polygon<T>::Optimize(float maxEps) {
		if (vertexes.size() <= 3)
			return;
		
		std::vector<Point<T>> newMyPoints;
		
		size_t curVertNum = 0;
		do {
			newMyPoints.push_back(vertexes[curVertNum]);
			curVertNum = GetNextOptimizeVertNumFromCurrent(curVertNum, maxEps);
		}
		while (curVertNum != -1);

		vertexes.clear();
		vertexes = newMyPoints;
	}

	template<class T>
	void Polygon<T>::Normalize(const Rect<T>& src, const Rect<T>& dst) {
		for (Point<T>& vertex : vertexes)
			vertex.NormalizeCoordinate(src, dst);
	}

	template<class T>
	bool Polygon<T>::IsFormOneIntersect(size_t startVertNum, size_t endVertNum, float maxEps) const {
		if (startVertNum < 0 || endVertNum < 0)
			return false;

		if (abs((float)startVertNum - endVertNum) <= 1.0f)
			return true;

		Point<T> startVert = vertexes[startVertNum];
		Point<T> endVert = vertexes[endVertNum];
		Point<T> checkVert;

		for (size_t checkVertNum = startVertNum + 1; checkVertNum < endVertNum - 1; checkVertNum++) {
			if (vertexes[checkVertNum].DistanceToLine(startVert, endVert) > maxEps)
				return false;
		}

		return true;
	}

	template<class T>
	size_t Polygon<T>::GetNextOptimizeVertNumFromCurrent(size_t currentNum, float maxEps) const {
		if (currentNum < 0 || currentNum >= vertexes.size() - 1)
			return -1;

		for (size_t nextVertNum = currentNum + 1; nextVertNum < vertexes.size(); nextVertNum++)
			if (!IsFormOneIntersect(currentNum, nextVertNum, maxEps))
				return nextVertNum - 1;

		return vertexes.size() - 1;
	}
	#pragma endregion

	#pragma endregion
}