#pragma once

namespace DataVisualization {
	class IPresentable {
	public:
		virtual void Draw() = 0;
		virtual void Impact(char keyCode, int button, int state, int x, int y) = 0;
	};
}