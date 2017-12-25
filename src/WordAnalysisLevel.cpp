#include <algorithm>
#include "../include/WordAnalysisLevel.h"

WordAnalysisLevel::WordAnalysisLevel() {
    inputLayer = new WordsInputLayer();
    layers.push_back(new MiddleLayer(NEURONS_1ST_LAYER, inputLayer->getOutput()->rows()));
    layers.push_back(new MiddleLayer(NEURONS_2ND_LAYER, NEURONS_1ST_LAYER));
    layers.push_back(new MiddleLayer(NEURONS_OUTPUT_LAYER, NEURONS_2ND_LAYER));
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

double WordAnalysisLevel::backpropagate(
	const std::vector<std::pair<std::vector<int>, Eigen::MatrixXf*>>& trainingExamples)
{
	float totalCost = 0.0f;

	std::vector<Eigen::MatrixXf*> weightsGradients;
	std::vector<Eigen::MatrixXf*> biasesGradients;

	weightsGradients.push_back(new Eigen::MatrixXf(NEURONS_1ST_LAYER, inputLayer->getOutput()->rows()));
	weightsGradients.push_back(new Eigen::MatrixXf(NEURONS_2ND_LAYER, NEURONS_1ST_LAYER));
	weightsGradients.push_back(new Eigen::MatrixXf(NEURONS_OUTPUT_LAYER, NEURONS_2ND_LAYER));

	biasesGradients.push_back(new Eigen::MatrixXf(NEURONS_1ST_LAYER, 1));
	biasesGradients.push_back(new Eigen::MatrixXf(NEURONS_2ND_LAYER, 1));
	biasesGradients.push_back(new Eigen::MatrixXf(NEURONS_OUTPUT_LAYER, 1));

	for (auto example : trainingExamples)
	{
		auto* output = analyzeWord(example.first);
		Eigen::MatrixXf gradient = (*output - *example.second);
		Eigen::MatrixXf input = *layers[LAYERS - 1]->getWeightedInput();
		
		atanDeriverate(input);

		Eigen::MatrixXf delta = gradient.cwiseProduct(input);

		totalCost += gradient.cwiseProduct(gradient).sum();

		for (int i = 0; i < LAYERS; i++)
		{

			if (i != LAYERS - 1)
			{
				delta = layers[LAYERS - 1 - i]->getWeights()->transpose() * delta;

				Eigen::MatrixXf inputGradient(*layers[LAYERS - 2 - i]->getWeightedInput());
				atanDeriverate(inputGradient);

				delta = delta.cwiseProduct(inputGradient);
			}
		}
	}

	for (auto e : weightsGradients)
		delete e;
	for (auto e : biasesGradients)
		delete e;

	totalCost /= (2 * trainingExamples.size());

	return totalCost;
}
