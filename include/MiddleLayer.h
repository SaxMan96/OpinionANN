#ifndef NEURAL_NETWORK_MIDDLELAYER_H
#define NEURAL_NETWORK_MIDDLELAYER_H

#include <Eigen>
#include "NeuronLayer.h"

//a hidden or output layer
class MiddleLayer : public NeuronLayer {
protected:
    Eigen::MatrixXf *connections;
    Eigen::MatrixXf *bias;
	Eigen::MatrixXf *weitghtedInput = nullptr;
	Eigen::MatrixXf *output;

	//arcus tangens normalised to <-1, 1>
	float activationFunction(float sum);
public:
    MiddleLayer(int neurons, int previousLayerNeurons);
    MiddleLayer(const MiddleLayer &middleLayer);
	//compute output basing on given output of the previous layer
    void computeOutput(Eigen::MatrixXf *previousOutput);
	//returns last output
    Eigen::MatrixXf* getOutput();
	//returns last output unmodified by activation function
	Eigen::MatrixXf* getWeightedInput();

	//returns connection weights
	Eigen::MatrixXf* getWeights();

	//initialises params with random values
    void initRandomConnections();

	//initialises params with given values
	//arg: { connections, biases }
	void initKnownConnections(const std::pair<Eigen::MatrixXf, Eigen::MatrixXf>& connections);
	//returns layer params
	//return value: { connections, biases }
	std::pair<Eigen::MatrixXf, Eigen::MatrixXf> getKnownConnections();

	//adds argument to current connection values
	void adjustConnections(Eigen::MatrixXf* diff);
	//adds argument to current biases values
	void adjustBiases(Eigen::MatrixXf* diff);

    ~MiddleLayer();
};


#endif //NEURAL_NETWORK_MIDDLELAYER_H
