#include "../include/NeuralNetwork.h"
#include <algorithm>
#include <thread>

NeuralNetwork::NeuralNetwork(InputLayer* inputLayer, std::vector<int> neuronsOnMiddleLayers) {
	layers.push_back(inputLayer);

	for (int i = 0; i < neuronsOnMiddleLayers.size(); i++)
		layers.push_back(new MiddleLayer(neuronsOnMiddleLayers[i], layers.back()->getNeuronNumber()));
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& network)
{
	layers.push_back(static_cast<InputLayer*>(network.layers[0])->newClone());

	for (int i = 1; i < network.layers.size(); i++)
		layers.push_back(new MiddleLayer(*static_cast<MiddleLayer*>(network.layers[i])));
}

NeuralNetwork::~NeuralNetwork() {
	for (auto* layer : layers)
		delete layer;
}

void NeuralNetwork::initRandomConnections() {
	for (int i = 1; i < layers.size(); i++)
		static_cast<MiddleLayer*>(layers[i])->initRandomConnections();
}

Eigen::MatrixXf* NeuralNetwork::computeOutput(Input* input) {
	static_cast<InputLayer*>(layers[0])->setInput(input);

	for (int i = 1; i < layers.size(); i++) {
		static_cast<MiddleLayer*>(layers[i])->computeOutput(layers[i - 1]->getOutput());
	}

	return layers.back()->getOutput();
}

inline float atanDeriverate(float f)
{
	return 1.0f / (1.0f + f * f);
}

inline static void atanDeriverate(Eigen::MatrixXf& X)
{
	for (int i = 0; i < X.rows(); i++)
		for (int j = 0; j < X.cols(); j++)
			X(i, j) = atanDeriverate(X(i, j));
}


void NeuralNetwork::calculateGradients(NeuralNetwork* network,
	std::vector<std::pair<Input*, Eigen::MatrixXf*>>* trainingExamples,
	std::vector<Eigen::MatrixXf*>* weightsGradients,
	std::vector<Eigen::MatrixXf*>* biasesGradients,
	double* totalCost)
{
	NeuralNetwork localCopy(*network);

	int layersN = localCopy.layers.size();

	for (auto example : *trainingExamples)
	{
		//calculate output
		Eigen::MatrixXf* output = localCopy.computeOutput(example.first);

		Eigen::MatrixXf gradient = (*output - *example.second);
		Eigen::MatrixXf input = *static_cast<MiddleLayer*>(localCopy.layers.back())->getWeightedInput();

		atanDeriverate(input);

		Eigen::MatrixXf delta = gradient.cwiseProduct(input);

		*totalCost += gradient.cwiseProduct(gradient).sum();

		for (int i = 0; i < layersN - 1; i++) //-1 for we don't adjust input layer
		{
			//propagate gradient from current layer to the previous one
			int prevLayer = layersN - 2 - i;

			Eigen::MatrixXf* prevLevelOutput = localCopy.layers[prevLayer]->getOutput();

			*(*weightsGradients)[prevLayer] += delta * prevLevelOutput->transpose();
			*(*biasesGradients)[prevLayer] += delta;

			if (i != layersN - 2) //calculate error for previous layer
			{
				delta = static_cast<MiddleLayer*>(localCopy.layers[prevLayer + 1])->getWeights()->transpose() * delta;

				Eigen::MatrixXf inputGradient(*static_cast<MiddleLayer*>(localCopy.layers[prevLayer])->getWeightedInput());
				atanDeriverate(inputGradient);

				delta = delta.cwiseProduct(inputGradient);
			}
		}
	}
}

double NeuralNetwork::backpropagate(
	const std::vector<std::pair<Input*, Eigen::MatrixXf*>>& trainingExamples,
	float learningSpeed, int maxThreads)
{
	double totalCost = 0.0;

	std::vector<std::vector<Eigen::MatrixXf*>*> weightsGradients;
	std::vector<std::vector<Eigen::MatrixXf*>*> biasesGradients;
	std::vector<std::vector<std::pair<Input*, Eigen::MatrixXf*>>*> examples;

	std::vector<double*> costs;
	std::vector<std::thread> threads;

	int threadsN = maxThreads < trainingExamples.size() ? maxThreads : trainingExamples.size();

	int examplesPerThread = trainingExamples.size() / threadsN;
	int threadsWithExtraExamples = trainingExamples.size() % threadsN;
	int example = 0;
	for (int i = 0; i < threadsN; i++)
	{
		int examplesInThread = examplesPerThread + (i < threadsWithExtraExamples);
		if (examplesInThread == 0)
			break; //next threads won't have more examples

		//prepare thread
		std::vector<Eigen::MatrixXf*>* threadWeightsGradients = new std::vector<Eigen::MatrixXf*>;
		std::vector<Eigen::MatrixXf*>* threadBiasesGradients = new std::vector<Eigen::MatrixXf*>;

		for (int j = 1; j < layers.size(); j++)
		{
			threadWeightsGradients->push_back(new Eigen::MatrixXf(layers[j]->getNeuronNumber(), layers[j - 1]->getNeuronNumber()));
			threadWeightsGradients->back()->setConstant(0);
			threadBiasesGradients->push_back(new Eigen::MatrixXf(layers[j]->getNeuronNumber(), 1));
			threadBiasesGradients->back()->setConstant(0);
		}

		auto* threadExamples = new std::vector<std::pair<Input*, Eigen::MatrixXf*>>;

		for (int j = 0; j < examplesInThread; j++)
			threadExamples->push_back(trainingExamples[example++]);

		double* threadCost = new double(0);

		weightsGradients.push_back(threadWeightsGradients);
		biasesGradients.push_back(threadBiasesGradients);
		examples.push_back(threadExamples);
		costs.push_back(threadCost);

		threads.push_back(std::thread(NeuralNetwork::calculateGradients,
			this, threadExamples, threadWeightsGradients, threadBiasesGradients, threadCost));
	}

	//wait until all threads stop their calculations
	for (auto& thread : threads)
		thread.join();

	for (double* cost : costs)
		totalCost += *cost;


	//adjust weights and biases

	for (int i = 0; i < layers.size() - 1; i++)
	{
		Eigen::MatrixXf total(layers[i + 1]->getNeuronNumber(), layers[i]->getNeuronNumber());
		total.setConstant(0);

		for (auto* e : weightsGradients)
			total += *(*e)[i];

		total *= -learningSpeed / (float)trainingExamples.size();
		static_cast<MiddleLayer*>(layers[i + 1])->adjustConnections(&total);
	}

	for (int i = 0; i < layers.size() - 1; i++)
	{
		Eigen::MatrixXf total(layers[i + 1]->getNeuronNumber(), 1);
		total.setConstant(0);

		for (auto* e : biasesGradients)
			total += *(*e)[i];

		total *= -learningSpeed / (float)trainingExamples.size();
		static_cast<MiddleLayer*>(layers[i + 1])->adjustBiases(&total);
	}

	//delete all temporary structures

	for (auto e : weightsGradients)
	{
		for (auto e_in : *e)
			delete e_in;
		delete e;
	}
	for (auto e : biasesGradients)
	{
		for (auto e_in : *e)
			delete e_in;
		delete e;
	}
	for (auto e : examples)
		delete e;
	for (auto e : costs)
		delete e;

	return totalCost / (2 * trainingExamples.size());
}


void NeuralNetwork::initKnownConnections(const std::vector<std::pair<Eigen::MatrixXf, Eigen::MatrixXf>>& connections)
{
	for (int i = 0; i < connections.size() && i + 1 < this->layers.size(); i++)
		static_cast<MiddleLayer*>(this->layers[i + 1])->initKnownConnections(connections[i]);
}

const std::vector<std::pair<Eigen::MatrixXf, Eigen::MatrixXf>> NeuralNetwork::getKnownConnections()
{
	std::vector<std::pair<Eigen::MatrixXf, Eigen::MatrixXf>> ret;

	for (int i = 1; i < layers.size(); i++)
		ret.push_back(static_cast<MiddleLayer*>(this->layers[i])->getKnownConnections());

	return ret;
}