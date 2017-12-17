#include "../include/OpinionAnalysisLevel.h"
#include <algorithm>

OpinionAnalysisLevel::OpinionAnalysisLevel() {
    inputLayer = new OpinionInputLayer();
    layers.push_back(new MiddleLayer(NEURONS_1ST_LAYER, inputLayer->getOutput()->rows()));
    layers.push_back(new MiddleLayer(NEURONS_2ND_LAYER, NEURONS_1ST_LAYER));
    layers.push_back(new MiddleLayer(NEURONS_OUTPUT_LAYER, NEURONS_2ND_LAYER));
}

OpinionAnalysisLevel::~OpinionAnalysisLevel() {
    for (auto it = layers.begin(); it < layers.end(); it++){
        delete(*it);
    }
    delete inputLayer;
}

void OpinionAnalysisLevel::initRandomConnections() {
    std::for_each(layers.begin(), layers.end(), [](MiddleLayer* layer){layer->initRandomConnections();});
}

Eigen::MatrixXf* OpinionAnalysisLevel::analyzeOpinion(std::vector<Eigen::MatrixXf> sentenceAnalysisResults,
                                                      std::vector<int> encodedSeparators) {
    inputLayer->computeOutput(sentenceAnalysisResults, encodedSeparators);
    layers[0]->computeOutput(inputLayer->getOutput());
    for (int i = 1; i < LAYERS; i++){
        layers[i]->computeOutput(layers[i-1]->getOutput());
    }
    return layers[LAYERS-1]->getOutput();
}
