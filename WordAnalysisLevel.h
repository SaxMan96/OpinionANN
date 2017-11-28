//
// Created by Delebrith on 24.11.2017.
//

#ifndef NEURAL_NETWORK_WORDANALYSISNETWORK_H
#define NEURAL_NETWORK_WORDANALYSISNETWORK_H


#include <vector>
#include "WordsLayer.h"
#include "WordsInputLayer.h"

class WordAnalysisLevel {
private:
    static const int LAYERS = 3; //input excluded, output included
    static const int NEURONS_1ST_LAYER = 600;
    static const int NEURONS_2ND_LAYER = 60;
    static const int NEURONS_OUTPUT_LAYER = 3;
    std::vector<WordsLayer *> layers;
    WordsInputLayer* inputLayer;
public:
    WordAnalysisLevel();
    void initRandomConnections();
    void initKnownConnections();
    Eigen::MatrixXf* analyzeWord(std::vector<int> encodedWord);
    ~WordAnalysisLevel();
};


#endif //NEURAL_NETWORK_WORDANALYSISNETWORK_H
