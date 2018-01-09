#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include "../include/WordAnalysisLevel.h"
#include "../include/UserInterface.h"


int main(int argc, char** argv)
{
	std::cout << "Hello, World! I am neural network" << std::endl;

	srand(time(NULL));

	argc--;
	argv++;

	if (argc == 2)
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