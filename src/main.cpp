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

	std::cout << "Witaj! Jestem siecią neuronową." << std::endl;

	srand(time(NULL));

	argc--;
	argv++;

	if (argc == 0)
	{
		std::cout << "Jak używać:" << std::endl
			 << "[plik z parametrami poziomu opinii] [plik z parametrami poziomu słów]" << std::endl
			 << "\t- Tryb interaktywny" << std::endl
			 << "[plik z parametrami poziomu słów] [plik z przypadkami treningowymi] [współczynnik uczenia] [liczba wątków]"
					 " [iteracje] " << std::endl
			 << "\t- Trening poziomu analizy słów" << std::endl
			 << "[plik z parametrami poziomu opinii] [plik z parametrami poziomu słów] [plik z przypadkami treningowymi]"
					 " [współczynnik uczenia] [liczba wątków] [iteracje] " << std::endl
			 << "\t- Trening poziomu analizy opinii" << std::endl;
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