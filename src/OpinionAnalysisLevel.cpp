#include "../include/OpinionAnalysisLevel.h"
#include "../include/OpinionInputLayer.h"

OpinionAnalysisLevel::OpinionAnalysisLevel() : NeuralNetwork(new OpinionInputLayer,
{ NEURONS_1ST_LAYER, NEURONS_2ND_LAYER, NEURONS_3RD_LAYER, NEURONS_OUTPUT_LAYER }) {
}