#include <iostream>
#include <ctime>
#include <vector>
#include "../include/WordAnalysisLevel.h"
#include "../include/SentenceInputLayer.h"
#include "../include/SentenceAnalysisLevel.h"
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

    SentenceInputLayer* sentenceInputLayer = new SentenceInputLayer();
    sentenceInputLayer->computeOutput(wordsAnalysisResult);
    std::cout << *(sentenceInputLayer->getOutput()) << std::endl << std::endl;

    SentenceAnalysisLevel* sentenceAnalysisLevel = new SentenceAnalysisLevel();
    sentenceAnalysisLevel->initRandomConnections();
    std::cout << *(sentenceAnalysisLevel->analyzeSentence(wordsAnalysisResult)) << std::endl << std::endl;

    std::vector<Eigen::MatrixXf> wordsAnalysisResult2;
    wordsAnalysisResult2.push_back(*(wordAnalysisLevel->analyzeWord(input2)));
    wordsAnalysisResult2.push_back(*(wordAnalysisLevel->analyzeWord(input)));

    std::cout << *(sentenceAnalysisLevel->analyzeSentence(wordsAnalysisResult2)) << std::endl << std::endl;

    std::vector<Eigen::MatrixXf> sentenceAnalisysResult;
    sentenceAnalisysResult.push_back(*(sentenceAnalysisLevel->analyzeSentence(wordsAnalysisResult)));
    sentenceAnalisysResult.push_back(*(sentenceAnalysisLevel->analyzeSentence(wordsAnalysisResult2)));

    std::vector<int> separators;
    separators.push_back(2);
    separators.push_back(1);

    OpinionInputLayer* opinionInputLayer = new OpinionInputLayer();
    opinionInputLayer->computeOutput(sentenceAnalisysResult, separators);
    std::cout << *(opinionInputLayer->getOutput()) << std::endl << std::endl;

    OpinionAnalysisLevel* opinionAnalysisLevel = new OpinionAnalysisLevel();
    opinionAnalysisLevel->initRandomConnections();
    std::cout << *(opinionAnalysisLevel->analyzeOpinion(sentenceAnalisysResult, separators)) << std::endl << std::endl;

    delete wordAnalysisLevel, sentenceAnalysisLevel, opinionInputLayer, opinionAnalysisLevel;

    return 0;
}