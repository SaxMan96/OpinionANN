#ifndef NEURAL_NETWORK_MIDDLELAYER_H
#define NEURAL_NETWORK_MIDDLELAYER_H

#include <Eigen>
#include "WordsInputLayer.h"

class MiddleLayer: public WordsInputLayer {
protected:
    int neurons;
    Eigen::MatrixXf *connections;
    Eigen::MatrixXf *bias;
	Eigen::MatrixXf *weitghtedInput = nullptr;
public:
    MiddleLayer(int neurons, int previousLayerNeurons);
    MiddleLayer(const MiddleLayer &middleLayer);
    void computeOutput(Eigen::MatrixXf *previousOutput);
    Eigen::MatrixXf* getOutput();
	Eigen::MatrixXf* getWeightedInput();

	Eigen::MatrixXf* getWeights();
    void initRandomConnections();
    void initKnownConnections();

	void adjustConnections(Eigen::MatrixXf* diff);
	void adjustBiases(Eigen::MatrixXf* diff);

    ~MiddleLayer();
};


#endif //NEURAL_NETWORK_MIDDLELAYER_H
