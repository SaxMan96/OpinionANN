#ifndef NEURAL_NETWORK_NEURALNETWORK_H
#define NEURAL_NETWORK_NEURALNETWORK_H


#include <string>
#include <vector>

class OpinionAnalysisNetwork {

    OpinionAnalysisNetwork();
    OpinionAnalysisNetwork(std::string);
    float compute(std::string input);
    void learn(std::string filename);
};


#endif //NEURAL_NETWORK_NEURALNETWORK_H
