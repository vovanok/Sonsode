#pragma once

#include <iostream>
#include <gl/glut.h>
#include <vector>
#include <string>
#include <map>
#include <functional>
#include "IPresentable.h"

namespace DataVisualization {
	namespace Graphic {
		using std::vector;
		using std::map;
		using std::function;
		using std::string;

		enum class CameraRotateDirection {
			Up,
			Down,
			Left,
			Right
		};

		enum class CameraZoomDirection {
			In,
			Out
		};

		class GraphicMgr {
		private:
			static GraphicMgr *Instance;

			static void OnDisplay();
			static void OnReshape(int, int);
			static void OnKeyboard(unsigned char, int, int);
			static void OnMouse(int, int, int, int);

			int wndHdlr;

			float camAngleShiftStep;
			float camHorzShift;
			float camVertShift;

			float zoomDim;
			float zoomStep;

			bool isPerspective;
			bool isLight;

			float leftVisibilityBound;
			float rightVisibilityBound;
			float upVisibilityBound;
			float downVisibilityBound;

			bool isRun;

			GraphicMgr();
			~GraphicMgr();
	
			void SetCamPosition();

			vector<IPresentable*> presentObjs;
			map<unsigned char, function<void()>> keyboardHandlers;
			function<void()> preDrawHandler;

			void RegisterDisplayFunc(void (*func)(void));
			void RegisterReshapeFunc(void (*func)(int, int));
			void RegisterKeyboardFunc(void (*func)(unsigned char, int, int));
			void RegisterMouseFunc(void (*func)(int, int, int, int));
	
			void onWindowDisplay();
			void onWindowReshape(int width, int height);
			void onWindowKeyboard(unsigned char key, int x, int y);
			void onWindowMouse(int button, int state, int x, int y);

			void Init(int argc, char **argv, const string windowName);
		public:
			static GraphicMgr *New(int argc, char **argv, const string windowName);
			static GraphicMgr *New(int argc, char **argv, const string windowName, bool isPerspective, bool isLight);
			static void Free();

			void ReDisplay();
			void Run();

			void StartAnimation();
			void StopAnimation();

			bool IsRunAnimation();

			void CameraRotate(CameraRotateDirection direction);
			void CameraZoom(CameraZoomDirection direction);

			void AddPresentObj(IPresentable *obj);
			void ClearPresentObjs();

			void AddUpdateKeyboardHandler(unsigned char key, function<void()> handler);
			void AddUpdateKeyboardHandler(string keys, function<void()> handler);
			void ClearKeyboardHandler();

			void SetVisibilityArea(float left, float right, float up, float down);

			void RegisterPreDrawHandler(function<void()> handler);
		};
	}
}