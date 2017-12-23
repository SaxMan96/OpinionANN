#ifndef NEURAL_NETWORK_MIDDLELAYER_H
#define NEURAL_NETWORK_MIDDLELAYER_H

#include <Eigen>
#include "WordsInputLayer.h"

class MiddleLayer: public WordsInputLayer {
protected:
    int neurons;
    Eigen::MatrixXf *connections;
	Eigen::MatrixXf *weitghtedInput = nullptr;
public:
    MiddleLayer(int neurons, int previousLayerNeurons);
    void computeOutput(Eigen::MatrixXf *previousOutput);
    Eigen::MatrixXf* getOutput();
	Eigen::MatrixXf* getWeightedInput();
    void initRandomConnections();
    void initKnownConnections();
    ~MiddleLayer();
};


#endif //NEURAL_NETWORK_MIDDLELAYER_H
