#ifndef NEURAL_NETWORK_INPUTLAYER_H
#define NEURAL_NETWORK_INPUTLAYER_H

#include <Eigen>
#include "NeuronLayer.h"
#include "Input.h"

//neuron layer for input
class InputLayer : public NeuronLayer
{
public:
	InputLayer(int neurons) : NeuronLayer(neurons) {}
	virtual void setInput(Input* input) = 0;

	//create a new-allocated clone
	virtual InputLayer* newClone() = 0;
};

#endif