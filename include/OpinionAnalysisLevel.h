#ifndef NEURAL_NETWORK_OPINIONANALISYSLEVEL_H
#define NEURAL_NETWORK_OPINIONANALISYSLEVEL_H


#include "../include/NeuralNetwork.h"

//class for opinion analysis
class OpinionAnalysisLevel : public NeuralNetwork {
public:
	static const int NEURONS_1ST_LAYER = 240;
	static const int NEURONS_2ND_LAYER = 120;
	static const int NEURONS_3RD_LAYER = 30;
	static const int NEURONS_OUTPUT_LAYER = 1;

    OpinionAnalysisLevel();
};


#endif //NEURAL_NETWORK_OPINIONANALISYSLEVEL_H
