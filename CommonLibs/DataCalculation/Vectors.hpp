#pragma once

template<class T> struct Vector2D {
	T x, y;
	Vector2D() { }
	Vector2D(T x, T y) {
		this->x = x;
		this->y = y;
	}
};

template<class T> struct Vector3D {
	T x, y, z;
	Vector3D() { }
	Vector3D(T x, T y, T z) {
		this->x = x;
		this->y = y;
		this->z = z;
	}
};