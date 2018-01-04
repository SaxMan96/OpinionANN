#ifndef NEURAL_NETWORK_WORDANALYSISNETWORK_H
#define NEURAL_NETWORK_WORDANALYSISNETWORK_H


#include <vector>
#include "MiddleLayer.h"
#include "WordsInputLayer.h"
#include <Eigen>

class WordAnalysisLevel {
private:
    static const int LAYERS = 3; //input excluded, output included
    static const int NEURONS_1ST_LAYER = 600;
    static const int NEURONS_2ND_LAYER = 60;
    std::vector<MiddleLayer *> layers;
    WordsInputLayer* inputLayer;
public:
    static const int NEURONS_OUTPUT_LAYER = 3;
    WordAnalysisLevel();
    WordAnalysisLevel(const WordAnalysisLevel &wordAnalysisLevel);
    void initRandomConnections();
    void initKnownConnections();
    Eigen::MatrixXf* analyzeWord(std::vector<int> encodedWord);
    ~WordAnalysisLevel();

	static void countGradients(WordAnalysisLevel* network,
		std::vector<std::pair<std::vector<int>, Eigen::MatrixXf*>>* trainingExamples,
		std::vector<Eigen::MatrixXf*>* weightsGradients,
		std::vector<Eigen::MatrixXf*>* biasesGradients,
		double* totalCost);

	double backpropagate(const std::vector<std::pair<std::vector<int>, Eigen::MatrixXf*>>& trainingExamples, float learningSpeed, int maxThreads);
};


#endif //NEURAL_NETWORK_WORDANALYSISNETWORK_H
