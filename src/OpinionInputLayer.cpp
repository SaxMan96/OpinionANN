#include <Eigen>
#include "../include/OpinionInputLayer.h"

OpinionInputLayer::OpinionInputLayer() {
    this->output = new Eigen::MatrixXf(this->MAX_SENTENCES_CONSIDERED * PARAMETERS_PER_SENTENCE +
                                               this->MAX_SENTENCES_CONSIDERED * this->SEPARATORS_POSSIBLE, 1);
    this->output->setZero();
}

Eigen::MatrixXf* OpinionInputLayer::getOutput() {
    return this->output;
}
/*
 * vector of separators must be the same size as sentence analysis results since correctly built opinion's pattern is:
 * sentence + separator + sentence + separator + ... + sentence + separator
 */
void OpinionInputLayer::computeOutput(std::vector<Eigen::MatrixXf> sentenceAnalysisResults,
                                      std::vector<int> separators) {
    this->output->setZero();
    for (int i = 0; i < sentenceAnalysisResults.size(); i++){
        for (int j = 0; j < this->PARAMETERS_PER_SENTENCE; j++){
            (*output)((this->PARAMETERS_PER_SENTENCE + this->SEPARATORS_POSSIBLE)*i + j, 0) =
                    sentenceAnalysisResults[i](j,0);
        }
        if (separators[i] != 0) {
            (*output)((this->PARAMETERS_PER_SENTENCE + this->SEPARATORS_POSSIBLE) * i
                      + PARAMETERS_PER_SENTENCE + separators[i] - 1, 0) = 1;
        }
    }
}

OpinionInputLayer::~OpinionInputLayer() {
    delete output;
}