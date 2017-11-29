//
// Created by Delebrith on 24.11.2017.
//

#ifndef NEURAL_NETWORK_WORDLAYER_H
#define NEURAL_NETWORK_WORDLAYER_H

#include <Eigen>

class WordsLayer {
private:
    int neurons;
    Eigen::MatrixXf *connections;
    Eigen::MatrixXf *output;
public:
    WordsLayer(int neurons, int previousLayerNeurons);
    void computeOutput(Eigen::MatrixXf *previousOutput);
    Eigen::MatrixXf* getOutput();
    void initRandomConnections();
    void initKnownConnections();
    ~WordsLayer();
};


#endif //NEURAL_NETWORK_WORDLAYER_H
