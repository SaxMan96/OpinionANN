#include <iostream>
#include <ctime>
#include "WordAnalysisLevel.h"

int main() {
    std::cout << "Hello, World! I am neural network" << std::endl;

    srand(time(NULL));

    std::vector<int> input;
    input.push_back(1);
    input.push_back(3);
    input.push_back(16);


    WordAnalysisLevel* wordAnalysisLevel = new WordAnalysisLevel();
    wordAnalysisLevel->initRandomConnections();
    std::cout << *wordAnalysisLevel->analyzeWord(input) << std::endl;

    delete wordAnalysisLevel;

    return 0;
}