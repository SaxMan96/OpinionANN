#include <iostream>
#include <math.h>
#include "../WordsLayer.h"

WordsLayer::WordsLayer(int neurons, int previousLayerNeurons) {
    this->neurons = neurons;
    this->connections = new Eigen::MatrixXf(neurons, previousLayerNeurons);
    this->connections->setOnes();
}

void WordsLayer::computeOutput(Eigen::MatrixXf* previousOutput) {

    Eigen::Product<Eigen::MatrixXf, Eigen::MatrixXf> product =  (*connections) * (*previousOutput);
    output = new Eigen::MatrixXf(product);
    for (int i = 0; i < output->cols(); i++){
        for (int j = 0; j < output->rows(); j++){
            (*output)(j, i) = atan((*output)(j, i))/(M_PI_2);
        }
    }
}

Eigen::MatrixXf* WordsLayer::getOutput(){
    return output;
}

void WordsLayer::initRandomConnections() {
    connections->setRandom();
}

WordsLayer::~WordsLayer() {
    if (connections != nullptr) delete connections;
    if (output != nullptr) delete output;
}

