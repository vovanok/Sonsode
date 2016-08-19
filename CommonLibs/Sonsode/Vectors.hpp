#pragma once

namespace Sonsode {
	template<class T>
	struct Vector2D {
		T x, y;
		Vector2D() { }
		Vector2D(T x, T y) : x(x), y(y) {
		}
	};

	template<class T>
	struct Vector3D {
		T x, y, z;
		Vector3D() { }
		Vector3D(T x, T y, T z) : x(x), y(y), z(z) {
		}
	};
}