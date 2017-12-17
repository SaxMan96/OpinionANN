#ifndef NEURAL_NETWORK_SENTENCEANALYSISLEVEL_H
#define NEURAL_NETWORK_SENTENCEANALYSISLEVEL_H

#include <vector>
#include <Eigen>
#include "../include/SentenceInputLayer.h"
#include "../include/MiddleLayer.h"

class SentenceAnalysisLevel {
private:
    static const int LAYERS = 3; //input excluded, output included
    static const int NEURONS_1ST_LAYER = 300;
    static const int NEURONS_2ND_LAYER = 60;
    std::vector<MiddleLayer *> layers;
    SentenceInputLayer* inputLayer;
public:
    static const int NEURONS_OUTPUT_LAYER = 3;
    SentenceAnalysisLevel();
    void initRandomConnections();
    void initKnownConnections();
    Eigen::MatrixXf* analyzeSentence(std::vector<Eigen::MatrixXf> wordAnalysisResults);
    ~SentenceAnalysisLevel();
};


#endif //NEURAL_NETWORK_SENTENCEANALYSISLEVEL_H
