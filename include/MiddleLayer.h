#ifndef NEURAL_NETWORK_MIDDLELAYER_H
#define NEURAL_NETWORK_MIDDLELAYER_H

#include <Eigen>
#include "WordsInputLayer.h"

class MiddleLayer: public WordsInputLayer {
protected:
    int neurons;
    Eigen::MatrixXf *connections;
    Eigen::MatrixXf *bias;
public:
    MiddleLayer(int neurons, int previousLayerNeurons);
    MiddleLayer(const MiddleLayer &middleLayer);
    void computeOutput(Eigen::MatrixXf *previousOutput);
    Eigen::MatrixXf* getOutput();
    void initRandomConnections();
    void initKnownConnections();
    ~MiddleLayer();
};


#endif //NEURAL_NETWORK_MIDDLELAYER_H
