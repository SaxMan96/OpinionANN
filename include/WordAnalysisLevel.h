#ifndef NEURAL_NETWORK_WORDANALYSISNETWORK_H
#define NEURAL_NETWORK_WORDANALYSISNETWORK_H


#include "NeuralNetwork.h"

//Neural network for word analysis
class WordAnalysisLevel : public NeuralNetwork {
public:
	static const int NEURONS_1ST_LAYER = 600;
	static const int NEURONS_2ND_LAYER = 60;
    static const int NEURONS_OUTPUT_LAYER = 3;

    WordAnalysisLevel();
};


#endif //NEURAL_NETWORK_WORDANALYSISNETWORK_H
