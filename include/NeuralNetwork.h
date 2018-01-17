#ifndef NEURAL_NETWORK_NEURALNETWORK_H
#define NEURAL_NETWORK_NEURALNETWORK_H

#include <vector>
#include <Eigen>
#include "MiddleLayer.h"
#include "InputLayer.h"

class NeuralNetwork {
private:
	std::vector<NeuronLayer*> layers;

	static void calculateGradients(NeuralNetwork* network,
		std::vector<std::pair<Input*, Eigen::MatrixXf*>>* trainingExamples,
		std::vector<Eigen::MatrixXf*>* weightsGradients,
		std::vector<Eigen::MatrixXf*>* biasesGradients,
		double* totalCost);
public:
	NeuralNetwork(InputLayer* inputLayer, std::vector<int> neuronsOnMiddleLayers);
	NeuralNetwork(const NeuralNetwork& network);

	void initRandomConnections();
	void initKnownConnections(const std::vector<std::pair<Eigen::MatrixXf, Eigen::MatrixXf>>& connections);
	const std::vector<std::pair<Eigen::MatrixXf, Eigen::MatrixXf>> getKnownConnections();

	Eigen::MatrixXf* computeOutput(Input* input);

	~NeuralNetwork();

	double backpropagate(
		const std::vector<std::pair<Input*, Eigen::MatrixXf*>>& trainingExamples,
		float learningSpeed, int maxThreads);
};

#endif