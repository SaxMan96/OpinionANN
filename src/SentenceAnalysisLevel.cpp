#include "../include/SentenceAnalysisLevel.h"
#include <algorithm>

SentenceAnalysisLevel::SentenceAnalysisLevel() {
    inputLayer = new SentenceInputLayer();
    layers.push_back(new MiddleLayer(NEURONS_1ST_LAYER, inputLayer->getOutput()->rows()));
    layers.push_back(new MiddleLayer(NEURONS_2ND_LAYER, NEURONS_1ST_LAYER));
    layers.push_back(new MiddleLayer(NEURONS_OUTPUT_LAYER, NEURONS_2ND_LAYER));
}

SentenceAnalysisLevel::~SentenceAnalysisLevel() {
    for (auto it = layers.begin(); it < layers.end(); it++){
        delete(*it);
    }
    delete inputLayer;
}

void SentenceAnalysisLevel::initRandomConnections() {
    std::for_each(layers.begin(), layers.end(), [](MiddleLayer* layer){layer->initRandomConnections();});
}

Eigen::MatrixXf* SentenceAnalysisLevel::analyzeSentence(std::vector<Eigen::MatrixXf> wordAnalysisResults) {
    inputLayer->computeOutput(wordAnalysisResults);
    layers[0]->computeOutput(inputLayer->getOutput());
    for (int i = 1; i < LAYERS; i++){
        layers[i]->computeOutput(layers[i-1]->getOutput());
    }
    return layers[LAYERS-1]->getOutput();
}
