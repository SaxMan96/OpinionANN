#include "../include/SentenceInputLayer.h"

SentenceInputLayer::SentenceInputLayer() {
    this->output = new Eigen::MatrixXf(this->WORDS_CONSIDERED * this->PARAMETERS_PER_WORD, 1);
    this->output->setZero();
}

Eigen::MatrixXf* SentenceInputLayer::getOutput() {
    return this->output;
}

void SentenceInputLayer::computeOutput(std::vector<Eigen::MatrixXf> wordAnalysisResults) {
    this->output->setZero();
    for (int i = 0; i < wordAnalysisResults.size(); i++){
        for (int j = 0; j < this->PARAMETERS_PER_WORD; j++) {
            (*output)(3*i + j, 0) = wordAnalysisResults[i](j, 0);
        }
    }
}

SentenceInputLayer::~SentenceInputLayer() {
    delete output;
}
