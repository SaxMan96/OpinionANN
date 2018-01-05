#ifndef NEURAL_NETWORK_OPINIONANALISYSLEVEL_H
#define NEURAL_NETWORK_OPINIONANALISYSLEVEL_H


#include <string>
#include <vector>
#include "../include/MiddleLayer.h"
#include "../include/OpinionInputLayer.h"

class OpinionAnalysisLevel {
private:
    static const int LAYERS = 4; //input excluded, output included
    static const int NEURONS_1ST_LAYER = 600;
    static const int NEURONS_2ND_LAYER = 300;
    static const int NEURONS_3RD_LAYER = 60;
    static const int NEURONS_OUTPUT_LAYER = 1;
    std::vector<MiddleLayer *> layers;
    OpinionInputLayer* inputLayer;
public:
    OpinionAnalysisLevel();
    OpinionAnalysisLevel(const OpinionAnalysisLevel &opinionAnalysisLevel);
    void initRandomConnections();
    void initKnownConnections();
    void addSentenceToInput(std::vector<Eigen::MatrixXf> wordsAnalysisResults);
    void resetInput();
    Eigen::MatrixXf* analyzeOpinion();
    ~OpinionAnalysisLevel();
};


#endif //NEURAL_NETWORK_OPINIONANALISYSLEVEL_H
