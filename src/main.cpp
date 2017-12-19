#include <iostream>
#include <ctime>
#include <vector>
#include "../include/WordAnalysisLevel.h"
#include "../include/OpinionInputLayer.h"
#include "../include/OpinionAnalysisLevel.h"

int main() {
    std::cout << "Hello, World! I am neural network" << std::endl;

    srand(time(NULL));

    std::vector<int> input;
    input.push_back(1);
    input.push_back(3);
    input.push_back(16);

    WordAnalysisLevel* wordAnalysisLevel = new WordAnalysisLevel();
    wordAnalysisLevel->initRandomConnections();
    std::cout << *(wordAnalysisLevel->analyzeWord(input)) << std::endl << std::endl;


    std::vector<int> input2;
    input2.push_back(4);
    input2.push_back(8);
    input2.push_back(11);
    input2.push_back(17);

    std::cout << *(wordAnalysisLevel->analyzeWord(input2)) << std::endl << std::endl;

    std::cout << std::endl;

    std::vector<Eigen::MatrixXf> wordsAnalysisResult;
    wordsAnalysisResult.push_back(*(wordAnalysisLevel->analyzeWord(input)));
    wordsAnalysisResult.push_back(*(wordAnalysisLevel->analyzeWord(input2)));

    std::vector<Eigen::MatrixXf> wordsAnalysisResult2;
    wordsAnalysisResult2.push_back(*(wordAnalysisLevel->analyzeWord(input2)));
    wordsAnalysisResult2.push_back(*(wordAnalysisLevel->analyzeWord(input)));

    OpinionAnalysisLevel* opinionAnalysisLevel = new OpinionAnalysisLevel();
    opinionAnalysisLevel->initRandomConnections();
    opinionAnalysisLevel->addSentenceToInput(wordsAnalysisResult);
    opinionAnalysisLevel->addSentenceToInput(wordsAnalysisResult2);
    std::cout << *(opinionAnalysisLevel->analyzeOpinion()) << std::endl << std::endl;

    delete wordAnalysisLevel, opinionAnalysisLevel;

    return 0;
}