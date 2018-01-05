#include "../include/OpinionAnalysisLevel.h"
#include <algorithm>

OpinionAnalysisLevel::OpinionAnalysisLevel() {
    inputLayer = new OpinionInputLayer();
    layers.push_back(new MiddleLayer(NEURONS_1ST_LAYER, inputLayer->getOutput()->rows()));
    layers.push_back(new MiddleLayer(NEURONS_2ND_LAYER, NEURONS_1ST_LAYER));
    layers.push_back(new MiddleLayer(NEURONS_3RD_LAYER, NEURONS_2ND_LAYER));
    layers.push_back(new MiddleLayer(NEURONS_OUTPUT_LAYER, NEURONS_3RD_LAYER));
}

OpinionAnalysisLevel::OpinionAnalysisLevel(const OpinionAnalysisLevel &opinionAnalysisLevel) {
    inputLayer = new OpinionInputLayer(*opinionAnalysisLevel.inputLayer);
    layers.push_back(new MiddleLayer(*opinionAnalysisLevel.layers[0]));
    layers.push_back(new MiddleLayer(*opinionAnalysisLevel.layers[1]));
    layers.push_back(new MiddleLayer(*opinionAnalysisLevel.layers[2]));
    layers.push_back(new MiddleLayer(*opinionAnalysisLevel.layers[3]));
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

void OpinionAnalysisLevel::addSentenceToInput(std::vector<Eigen::MatrixXf> wordsAnalysisResults){
    this->inputLayer->addSentence(wordsAnalysisResults);
}

void OpinionAnalysisLevel::resetInput() {
    this->inputLayer->reset();
}

Eigen::MatrixXf* OpinionAnalysisLevel::analyzeOpinion() {
    layers[0]->computeOutput(inputLayer->getOutput());
    for (int i = 1; i < LAYERS; i++){
        layers[i]->computeOutput(layers[i-1]->getOutput());
    }
    return layers[LAYERS-1]->getOutput();
}
