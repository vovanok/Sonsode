#include <iostream>
//#include "TimeMeter.h"
#include "HostData2D.hpp"
#include "GraphicMgr.h"
#include "IPresentable.h"

using namespace std;

class ProbaPresentable : public IPresentable
{
public:
	virtual void Draw()
	{}

	virtual void Impact(char keyCode, int button, int state, int x, int y)
	{}
};

void HostData2DTest()
{
	int dimX = 10;
	int dimY = 10;

	HostData2D<float>* proba = HostData2D<float>::New(dimX, dimY, 5);

	for(int y = 0; y < dimY; y++)
	{
		for (int x = 0; x < dimX; x++)
		{
			if (x == y)
				proba->at(x, y) = 2;
		}
	}

	for(int y = 0; y < dimY; y++)
	{
		for (int x = 0; x < dimX; x++)
		{
			cout << proba->at(x, y) << " ";
		}
		cout << endl;
	}

	cout << "HostData2DTest complete" << endl;
}

void GraphicMgrTest(int argc, char** argv)
{
	GraphicMgr *gr = GraphicMgr::New(argc, argv, "test");
	ProbaPresentable* prPresent = new ProbaPresentable();
	gr->AddPresentObj(dynamic_cast<IPresentable*>(prPresent));

	cout << "GraphicMgrTest complete" << endl;
}

void main(int argc, char** argv)
{
	HostData2DTest();

	GraphicMgrTest(argc, argv);

	cout << "Press any key to continue..." << endl;
	getchar();
}