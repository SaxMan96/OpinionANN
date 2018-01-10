#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <Windows.h>
#include "../include/WordAnalysisLevel.h"
#include "../include/UserInterface.h"


int main(int argc, char** argv)
{
	setlocale(LC_ALL, "PL_pl.UTF-8");
	SetConsoleOutputCP(CP_UTF8);
	SetConsoleCP(CP_UTF8);

	std::cout << "Hello, World! I am neural network" << std::endl;

	srand(time(NULL));

	argc--;
	argv++;

	if (argc == 0)
	{
		std::cout << "Usage:" << std::endl
			 << "[opinion network file] [word network file]" << std::endl
			 << "\t- Interactive mode" << std::endl
			 << "[word network file] [test examples file] [learning factor] [max threads] [iterations] " << std::endl
			 << "\t- Teach word network" << std::endl
			 << "[opinion network file] [word network file] [test examples file] [learning factor] [max threads] [iterations] " << std::endl
			 << "\t- Teach opinion network" << std::endl;
	}
	else if (argc == 2)
	{
		UserInterface::activateInteractions(argv);
	}
	else if (argc == 5)
	{
		UserInterface::performWordLevelTraining(argv);

	}
	else if (argc == 6)
	{
		UserInterface::performOpinionLevelTraining(argv);
	}


	return 0;
}