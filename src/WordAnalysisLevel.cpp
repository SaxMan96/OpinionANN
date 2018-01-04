#include <algorithm>
#include <thread>
#include "../include/WordAnalysisLevel.h"

WordAnalysisLevel::WordAnalysisLevel() {
    inputLayer = new WordsInputLayer();
    layers.push_back(new MiddleLayer(NEURONS_1ST_LAYER, inputLayer->getOutput()->rows()));
    layers.push_back(new MiddleLayer(NEURONS_2ND_LAYER, NEURONS_1ST_LAYER));
    layers.push_back(new MiddleLayer(NEURONS_OUTPUT_LAYER, NEURONS_2ND_LAYER));
}

WordAnalysisLevel::WordAnalysisLevel(const WordAnalysisLevel &wordAnalysisLevel) {
    inputLayer = new WordsInputLayer(*wordAnalysisLevel.inputLayer);
    layers.push_back(new MiddleLayer(*wordAnalysisLevel.layers[0]));
    layers.push_back(new MiddleLayer(*wordAnalysisLevel.layers[1]));
    layers.push_back(new MiddleLayer(*wordAnalysisLevel.layers[2]));
}

WordAnalysisLevel::~WordAnalysisLevel() {
    for (auto it = layers.begin(); it < layers.end(); it++){
        delete(*it);
    }
    delete inputLayer;
}

void WordAnalysisLevel::initRandomConnections() {
    std::for_each(layers.begin(), layers.end(), [](MiddleLayer* layer){layer->initRandomConnections();});
}

Eigen::MatrixXf* WordAnalysisLevel::analyzeWord(std::vector<int> encodedWord) {
    inputLayer->computeOutput(encodedWord);
    layers[0]->computeOutput(inputLayer->getOutput());
    for (int i = 1; i < LAYERS; i++){
        layers[i]->computeOutput(layers[i-1]->getOutput());
    }
    return layers[LAYERS-1]->getOutput();
}

inline float atanDeriverate(float f)
{
	return 1.0f / (1.0f + f * f);
}

inline void atanDeriverate(Eigen::MatrixXf& X)
{
	for (int i = 0; i < X.rows(); i++)
		for (int j = 0; j < X.cols(); j++)
			X(i, j) = atanDeriverate(X(i, j));
}

void WordAnalysisLevel::countGradients(WordAnalysisLevel* network,
	std::vector<std::pair<std::vector<int>, Eigen::MatrixXf*>>* trainingExamples,
	std::vector<Eigen::MatrixXf*>* weightsGradients,
	std::vector<Eigen::MatrixXf*>* biasesGradients,
	double* totalCost)
{
	WordAnalysisLevel localCopy(*network);

	for (auto example : *trainingExamples)
	{
		auto* output = localCopy.analyzeWord(example.first);
		Eigen::MatrixXf gradient = (*output - *example.second);
		Eigen::MatrixXf input = *localCopy.layers[LAYERS - 1]->getWeightedInput();

		atanDeriverate(input);

		Eigen::MatrixXf delta = gradient.cwiseProduct(input);

		*totalCost += gradient.cwiseProduct(gradient).sum();

		for (int i = 0; i < LAYERS; i++)
		{
			Eigen::MatrixXf* prevLevelOutput = i == LAYERS - 1 ? localCopy.inputLayer->getOutput() : localCopy.layers[LAYERS - 2 - i]->getOutput();

			*(*weightsGradients)[LAYERS - 1 - i] += delta * prevLevelOutput->transpose();
			*(*biasesGradients)[LAYERS - 1 - i] += delta;

			if (i != LAYERS - 1) //count error for previous layer
			{
				delta = localCopy.layers[LAYERS - 1 - i]->getWeights()->transpose() * delta;

				Eigen::MatrixXf inputGradient(*localCopy.layers[LAYERS - 2 - i]->getWeightedInput());
				atanDeriverate(inputGradient);

				delta = delta.cwiseProduct(inputGradient);
			}
		}
	}
}

double WordAnalysisLevel::backpropagate(
	const std::vector<std::pair<std::vector<int>, Eigen::MatrixXf*>>& trainingExamples, float learningSpeed, int maxThreads)
{
	double totalCost = 0.0;

	std::vector<std::vector<Eigen::MatrixXf*>*> weightsGradients;
	std::vector<std::vector<Eigen::MatrixXf*>*> biasesGradients;
	std::vector<std::vector<std::pair<std::vector<int>, Eigen::MatrixXf*>>*> examples;

	std::vector<double*> costs;
	std::vector<std::thread> threads;

	int threadsN = maxThreads < trainingExamples.size() ? maxThreads : trainingExamples.size();

	int examplesPerThread = trainingExamples.size() / threadsN;
	int threadsWithExtraExamples = trainingExamples.size() % threadsN;
	int example = 0;
	for (int i = 0; i < threadsN; i++)
	{
		//prepare thread
		std::vector<Eigen::MatrixXf*>* w = new std::vector<Eigen::MatrixXf*>;
		std::vector<Eigen::MatrixXf*>* b = new std::vector<Eigen::MatrixXf*>;

		w->push_back(new Eigen::MatrixXf(NEURONS_1ST_LAYER, inputLayer->getOutput()->rows()));
		w->push_back(new Eigen::MatrixXf(NEURONS_2ND_LAYER, NEURONS_1ST_LAYER));
		w->push_back(new Eigen::MatrixXf(NEURONS_OUTPUT_LAYER, NEURONS_2ND_LAYER));

		for (auto e : *w)
			e->setConstant(0);

		b->push_back(new Eigen::MatrixXf(NEURONS_1ST_LAYER, 1));
		b->push_back(new Eigen::MatrixXf(NEURONS_2ND_LAYER, 1));
		b->push_back(new Eigen::MatrixXf(NEURONS_OUTPUT_LAYER, 1));

		for (auto e : *b)
			e->setConstant(0);

		std::vector<std::pair<std::vector<int>, Eigen::MatrixXf*>>* e = new std::vector<std::pair<std::vector<int>, Eigen::MatrixXf*>>;

		for (int j = 0; j < examplesPerThread || j == examplesPerThread && j < threadsWithExtraExamples; j++)
			e->push_back(trainingExamples[example++]);

		double* c = new double(0);

		weightsGradients.push_back(w);
		biasesGradients.push_back(b);
		examples.push_back(e);
		costs.push_back(c);

		threads.push_back(std::thread(WordAnalysisLevel::countGradients, this, e, w, b, c));
	}

	//wait until all threads stop their calculations
	for (auto& e : threads)
		e.join();

	for (double* cost : costs)
		totalCost += *cost;


	//adjust weights and biases

	for (int i = 0; i < layers.size(); i++)
	{
		Eigen::MatrixXf total(layers[i]->getWeights()->rows(), layers[i]->getWeights()->cols());
		total.setConstant(0);

		for (auto* e : weightsGradients)
			total += *(*e)[i];

		total *= -learningSpeed / (float)trainingExamples.size();
		layers[i]->adjustConnections(&total);
	}

	for (int i = 0; i < layers.size(); i++)
	{
		Eigen::MatrixXf total(layers[i]->getOutput()->rows(), layers[i]->getOutput()->cols());
		total.setConstant(0);

		for (auto* e : biasesGradients)
			total += *(*e)[i];

		total *= -learningSpeed / (float)trainingExamples.size();
		layers[i]->adjustBiases(&total);
	}

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
