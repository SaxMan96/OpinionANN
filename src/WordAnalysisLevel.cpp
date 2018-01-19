#include "../include/WordAnalysisLevel.h"
#include "../include/WordsInputLayer.h"

WordAnalysisLevel::WordAnalysisLevel() : NeuralNetwork(new WordsInputLayer,
{ NEURONS_1ST_LAYER, NEURONS_2ND_LAYER, NEURONS_OUTPUT_LAYER }) {
}