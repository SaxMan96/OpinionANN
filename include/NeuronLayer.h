#ifndef NEURAL_NETWORK_NEURONLAYER_H
#define NEURAL_NETWORK_NEURONLAYER_H

#include <Eigen>

class NeuronLayer
{
public:
	NeuronLayer(int neurons);
	int getNeuronNumber() const;
	virtual Eigen::MatrixXf* getOutput() = 0;
private:
	int neurons;
};

#endif