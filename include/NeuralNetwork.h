#ifndef NEURAL_NETWORK_NEURALNETWORK_H
#define NEURAL_NETWORK_NEURALNETWORK_H

#include <vector>
#include <Eigen>
#include "MiddleLayer.h"
#include "InputLayer.h"

//class for neural network
class NeuralNetwork {
private:
	std::vector<NeuronLayer*> layers;

	//calculates total gradients for given training examples
	static void calculateGradients(NeuralNetwork* network,
		std::vector<std::pair<Input*, Eigen::MatrixXf*>>* trainingExamples,
		std::vector<Eigen::MatrixXf*>* weightsGradients,
		std::vector<Eigen::MatrixXf*>* biasesGradients,
		double* totalCost);
public:
	NeuralNetwork(InputLayer* inputLayer, std::vector<int> neuronsOnMiddleLayers);
	NeuralNetwork(const NeuralNetwork& network);

	//initialises random weights and biases in all layers
	void initRandomConnections();
	//sets weights and biases in all layers to given values
	void initKnownConnections(const std::vector<std::pair<Eigen::MatrixXf, Eigen::MatrixXf>>& connections);
	//returns weights and biases for all layers
	const std::vector<std::pair<Eigen::MatrixXf, Eigen::MatrixXf>> getKnownConnections();

	Eigen::MatrixXf* computeOutput(Input* input);

	~NeuralNetwork();

	//trains network
	//single iteration with given learning speed on given examples
	//divides process up to given number of threads (less if not enough training data)
	double backpropagate(
		const std::vector<std::pair<Input*, Eigen::MatrixXf*>>& trainingExamples,
		float learningSpeed, int maxThreads);
};

#endif