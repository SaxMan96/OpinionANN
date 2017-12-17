#ifndef NEURAL_NETWORK_OPINIONANALISYSLEVEL_H
#define NEURAL_NETWORK_OPINIONANALISYSLEVEL_H


#include <string>
#include <vector>
#include "../include/MiddleLayer.h"
#include "../include/OpinionInputLayer.h"

class OpinionAnalysisLevel {
private:
    static const int LAYERS = 3; //input excluded, output included
    static const int NEURONS_1ST_LAYER = 600;
    static const int NEURONS_2ND_LAYER = 60;
    static const int NEURONS_OUTPUT_LAYER = 1;
    std::vector<MiddleLayer *> layers;
    OpinionInputLayer* inputLayer;
public:
    OpinionAnalysisLevel();
    void initRandomConnections();
    void initKnownConnections();
    Eigen::MatrixXf* analyzeOpinion(std::vector<Eigen::MatrixXf> sentenceAnalysisResults,
                                    std::vector<int> encodedSeparators);
    ~OpinionAnalysisLevel();
};


#endif //NEURAL_NETWORK_OPINIONANALISYSLEVEL_H
