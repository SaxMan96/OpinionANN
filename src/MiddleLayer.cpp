#define _USE_MATH_DEFINES

#include <iostream>
#include <math.h>
#include "../include/MiddleLayer.h"

MiddleLayer::MiddleLayer(int neurons, int previousLayerNeurons) {
    this->neurons = neurons;
    this->connections = new Eigen::MatrixXf(neurons, previousLayerNeurons);
    this->bias = new Eigen::MatrixXf(neurons, 1);
    this->connections->setOnes();
}

void MiddleLayer::computeOutput(Eigen::MatrixXf* previousOutput) {
    if (output != nullptr) delete output;
    Eigen::Product<Eigen::MatrixXf, Eigen::MatrixXf> product =  (*connections) * (*previousOutput);
    output = new Eigen::MatrixXf(product);
    for (int i = 0; i < output->cols(); i++){
        for (int j = 0; j < output->rows(); j++){
            (*output)(j, i) = atan((*output)(j, i) - (*bias)(j, i))/(M_PI_2);
        }
    }
}

Eigen::MatrixXf* MiddleLayer::getOutput(){
    return output;
}

void MiddleLayer::initRandomConnections() {
    connections->setRandom();
    bias->setRandom();
}

MiddleLayer::~MiddleLayer() {
    if (connections != nullptr) delete connections;
    if (output != nullptr) delete output;
}

