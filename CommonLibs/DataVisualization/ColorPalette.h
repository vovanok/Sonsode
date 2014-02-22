#pragma once

#include <map>

struct Color {
	float r, g, b, a;
	Color();
	Color(float r, float g, float b, float a);
private:
	void Init(float r, float g, float b, float a);
};

class ColorPalette {
public:
	ColorPalette(Color beginColor, Color endColor, float beginValue, float endValue);
	void AddUpdControlColor(size_t value, Color color);
	Color GetColor(float value) const;
	Color GetColor(size_t value) const;
	~ColorPalette();

private:
	std::map<size_t, Color> controlColors;
	size_t minValue;
	size_t maxValue;
	float beginValue, endValue;
};