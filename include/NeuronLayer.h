#ifndef NEURAL_NETWORK_NEURONLAYER_H
#define NEURAL_NETWORK_NEURONLAYER_H

#include <Eigen>

//a single layer of neurons
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