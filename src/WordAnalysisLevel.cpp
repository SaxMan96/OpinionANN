#include <algorithm>
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

