#include "..\include\NeuronLayer.h"

NeuronLayer::NeuronLayer(int neurons)
{
	this->neurons = neurons;
}

int NeuronLayer::getNeuronNumber() const
{
	return this->neurons;
}