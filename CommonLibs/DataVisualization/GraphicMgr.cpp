#include "GraphicMgr.h"

void EmptyFunc() {}

void GraphicMgr::OnDisplay() {
	GraphicMgr::Instance->onWindowDisplay();
}

void GraphicMgr::OnReshape(int width, int height) {
	GraphicMgr::Instance->onWindowReshape(width, height);
}

void GraphicMgr::OnKeyboard(unsigned char key, int x, int y) {
	GraphicMgr::Instance->onWindowKeyboard(key, x, y);
}

void GraphicMgr::OnMouse(int button, int state, int x, int y) {
	GraphicMgr::Instance->onWindowMouse(button, state, x, y);
}

GraphicMgr *GraphicMgr::Instance = 0;
GraphicMgr *GraphicMgr::New(int argc, char **argv, const std::string windowName) {
	return New(argc,argv, windowName, true, false);
}

GraphicMgr *GraphicMgr::New(int argc, char **argv, const std::string windowName, bool isPerspective, bool isLight) {
	if (GraphicMgr::Instance == 0) {
		GraphicMgr::Instance = new GraphicMgr();
		GraphicMgr::Instance->isPerspective = isPerspective;
		GraphicMgr::Instance->isLight = isLight;
		GraphicMgr::Instance->Init(argc, argv, windowName);
		GraphicMgr::Instance->RegisterDisplayFunc(OnDisplay);
		GraphicMgr::Instance->RegisterReshapeFunc(OnReshape);
		GraphicMgr::Instance->RegisterKeyboardFunc(OnKeyboard);
		GraphicMgr::Instance->RegisterMouseFunc(OnMouse);
	}
	return GraphicMgr::Instance;
}

void GraphicMgr::Free() {
	if (GraphicMgr::Instance != 0)
		delete GraphicMgr::Instance;
}

//constructor & destructor
GraphicMgr::GraphicMgr() {
	wndHdlr = 0;
	SetVisibilityArea(-30, 30, 30, -30);//-20, 20, 20, -20);
}

GraphicMgr::~GraphicMgr() {
	if (this->wndHdlr != 0)
		glutDestroyWindow(wndHdlr);
}

//Init
void GraphicMgr::Init(int argc, char **argv, const std::string windowName) {
	glutInit (&argc, argv);
	glutInitWindowPosition (200, 50);
	glutInitWindowSize(800, 800);
	glutInitDisplayMode (GLUT_RGBA | GLUT_SINGLE);
	
	wndHdlr = glutCreateWindow(windowName.c_str());
	
	camAngleShiftStep = 5;
	zoomDim = -30;
	zoomStep = 1;
	camHorzShift = 0;
	camVertShift = 0;
	isRun = false;
	RegisterPreDrawHandler(EmptyFunc);

	glClearColor(1, 1, 1, 1);//0, 0, 0, 1);///!!!

	glEnable(GL_DEPTH_TEST);

	if (isLight) {
		glEnable(GL_LIGHTING);
		float white_light[] = {1.0f, 1.0f, 1.0f};
		float ambient[] = {0.5f, 0.5f, 0.5f};
		glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient);//white_light);

		float light_position[] = {0, 0, -50};
		glEnable(GL_LIGHT0);
		glLightfv(GL_LIGHT0, GL_POSITION, light_position);
		glLightfv(GL_LIGHT0,GL_AMBIENT, white_light);
		glLightfv(GL_LIGHT0, GL_DIFFUSE, white_light);
		glLightfv(GL_LIGHT0, GL_SPECULAR, white_light);
		//glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 5);
		//glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0);
		//glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 0);
	}
}
//API
void GraphicMgr::ReDisplay() {
	glutPostRedisplay();
}

void GraphicMgr::onWindowDisplay() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(0, 0, this->zoomDim);
	SetCamPosition();

	preDrawHandler();

	for(size_t i = 0; i < presentObjs.size(); i++) 	{
		presentObjs[i]->Draw();
	}
	
	glFlush();

	if (isRun)
		glutPostRedisplay();
}

void GraphicMgr::onWindowReshape(int width, int height) {
	glViewport(0, 0, (GLsizei)width, (GLsizei)height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	if (isPerspective) {
		glFrustum(leftVisibilityBound, rightVisibilityBound,
			downVisibilityBound, upVisibilityBound, 0, 200);
	} else {
		glOrtho(leftVisibilityBound, rightVisibilityBound,
			downVisibilityBound, upVisibilityBound, 0, 200);
	}

	glMatrixMode(GL_MODELVIEW);
}

void GraphicMgr::onWindowKeyboard(unsigned char key, int x, int y) {
	keyboardHandlers[key]();

	for(size_t i = 0; i < presentObjs.size(); i++)
		presentObjs[i]->Impact(key, 0, 0, x, y);

	ReDisplay();
}

void GraphicMgr::onWindowMouse(int button, int state, int x, int y) {
}

void GraphicMgr::RegisterDisplayFunc(void (*func)(void)) {
	glutDisplayFunc(func);
}

void GraphicMgr::RegisterReshapeFunc(void (*func)(int, int)) {
	glutReshapeFunc(func);
}

void GraphicMgr::RegisterKeyboardFunc(void (*func)(unsigned char, int, int)) {
	glutKeyboardFunc(func);
}

void GraphicMgr::RegisterMouseFunc(void (*func)(int, int, int, int)) {
	glutMouseFunc(func);
}

void GraphicMgr::Run() {
	glutMainLoop();
}

void GraphicMgr::CameraRotate(CameraRotateDirection direction) {
	switch(direction) {
		case CameraRotateDirection::Up:
			this->camVertShift += this->camAngleShiftStep;
			break;
		case CameraRotateDirection::Down:
			this->camVertShift -= this->camAngleShiftStep;
			break;
		case CameraRotateDirection::Left:
			this->camHorzShift += this->camAngleShiftStep;
			break;
		case CameraRotateDirection::Right:
			this->camHorzShift -= this->camAngleShiftStep;
			break;
	}
}

void GraphicMgr::CameraZoom(CameraZoomDirection direction) {
	switch(direction) {
		case CameraZoomDirection::In:
			this->zoomDim += this->zoomStep;
			break;
		case CameraZoomDirection::Out:
			this->zoomDim -= this->zoomStep;
			break;
	}
}

void GraphicMgr::SetCamPosition() {
	glRotatef(this->camVertShift, 1, 0, 0);
	glRotatef(this->camHorzShift, 0, 1, 0);
}

void GraphicMgr::AddPresentObj(IPresentable *obj) {
	presentObjs.push_back(obj);
}

void GraphicMgr::ClearPresentObjs() {
	presentObjs.clear();
}

void GraphicMgr::AddUpdateKeyboardHandler(unsigned char key, std::function<void()> handler) {
	keyboardHandlers[key] = handler;
}

void GraphicMgr::AddUpdateKeyboardHandler(std::string keys, std::function<void()> handler) {
	for(char key : keys)
		AddUpdateKeyboardHandler(key, handler);
}

void GraphicMgr::ClearKeyboardHandler() {
	keyboardHandlers.clear();
}

void GraphicMgr::SetVisibilityArea(float left, float right, float up, float down) {
	leftVisibilityBound = left;
	rightVisibilityBound = right;
	upVisibilityBound = up;
	downVisibilityBound = down;
}

void GraphicMgr::StartAnimation() {
	isRun = true;
	ReDisplay();
}

void GraphicMgr::StopAnimation() {
	isRun = false;
}

bool GraphicMgr::IsRunAnimation() {
	return isRun;
}

void GraphicMgr::RegisterPreDrawHandler(std::function<void()> handler) {
	preDrawHandler = handler;
}