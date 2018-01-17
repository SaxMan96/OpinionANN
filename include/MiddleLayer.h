#ifndef NEURAL_NETWORK_MIDDLELAYER_H
#define NEURAL_NETWORK_MIDDLELAYER_H

#include <Eigen>
#include "NeuronLayer.h"

class MiddleLayer : public NeuronLayer {
protected:
    int neurons;
    Eigen::MatrixXf *connections;
    Eigen::MatrixXf *bias;
	Eigen::MatrixXf *weitghtedInput = nullptr;
	Eigen::MatrixXf *output;
public:
    MiddleLayer(int neurons, int previousLayerNeurons);
    MiddleLayer(const MiddleLayer &middleLayer);
    void computeOutput(Eigen::MatrixXf *previousOutput);
    Eigen::MatrixXf* getOutput();
	Eigen::MatrixXf* getWeightedInput();

	Eigen::MatrixXf* getWeights();
    void initRandomConnections();

	void initKnownConnections(const std::pair<Eigen::MatrixXf, Eigen::MatrixXf>& connections);
	std::pair<Eigen::MatrixXf, Eigen::MatrixXf> getKnownConnections();

	void adjustConnections(Eigen::MatrixXf* diff);
	void adjustBiases(Eigen::MatrixXf* diff);

    ~MiddleLayer();
};


#endif //NEURAL_NETWORK_MIDDLELAYER_H
