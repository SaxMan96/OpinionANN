#ifndef NEURAL_NETWORK_OPINIONANALISYSLEVEL_H
#define NEURAL_NETWORK_OPINIONANALISYSLEVEL_H


#include <string>
#include <vector>
#include "../include/MiddleLayer.h"
#include "../include/OpinionInputLayer.h"

class OpinionAnalysisLevel {
private:
    std::vector<MiddleLayer *> layers;
    OpinionInputLayer* inputLayer;

	static void calculateGradients(OpinionAnalysisLevel* network,
		std::vector<std::pair<std::vector<std::vector<Eigen::MatrixXf>>, Eigen::MatrixXf*>>* trainingExamples,
		std::vector<Eigen::MatrixXf*>* weightsGradients,
		std::vector<Eigen::MatrixXf*>* biasesGradients,
		double* totalCost);
public:
	static const int LAYERS = 4; //input excluded, output included
	static const int NEURONS_1ST_LAYER = 600;
	static const int NEURONS_2ND_LAYER = 300;
	static const int NEURONS_3RD_LAYER = 60;
	static const int NEURONS_OUTPUT_LAYER = 1;

    OpinionAnalysisLevel();
    OpinionAnalysisLevel(const OpinionAnalysisLevel &opinionAnalysisLevel);
    void initRandomConnections();
	void initKnownConnections(const std::vector<std::pair<Eigen::MatrixXf, Eigen::MatrixXf>>& connections);
	const std::vector<std::pair<Eigen::MatrixXf, Eigen::MatrixXf>> getKnownConnections();
    void addSentenceToInput(std::vector<Eigen::MatrixXf> wordsAnalysisResults);
    void resetInput();
    Eigen::MatrixXf* analyzeOpinion();
    ~OpinionAnalysisLevel();

	double backpropagate(
		const std::vector<std::pair<std::vector<std::vector<Eigen::MatrixXf>>, Eigen::MatrixXf*>>& trainingExamples,
		float learningSpeed, int maxThreads);
};


#endif //NEURAL_NETWORK_OPINIONANALISYSLEVEL_H
