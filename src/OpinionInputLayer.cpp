#include <Eigen>
#include "../include/OpinionInputLayer.h"

OpinionInputLayer::OpinionInputLayer() {
    this->output = new Eigen::MatrixXf(this->MAX_SENTENCES_CONSIDERED * this->MAX_WORDS_CONSIDERED
                                       * this->PARAMETERS_PER_WORD, 1);
    this->output->setZero();
}

OpinionInputLayer::OpinionInputLayer(const OpinionInputLayer &opinionInputLayer) {
    this->output = new Eigen::MatrixXf(*opinionInputLayer.output);
}

Eigen::MatrixXf* OpinionInputLayer::getOutput() {
    return this->output;
}

void OpinionInputLayer::addSentence(std::vector<Eigen::MatrixXf> wordsAnalysisLevel){
    if (this->sentenceCounter >= this->MAX_SENTENCES_CONSIDERED) return;
    for (int i = 0; i < this->MAX_WORDS_CONSIDERED && i < wordsAnalysisLevel.size(); i++){
        for (int j = 0; j < this->PARAMETERS_PER_WORD; j++){
            int index = (this->sentenceCounter*this->MAX_WORDS_CONSIDERED) + (i*this->PARAMETERS_PER_WORD) + j;
            (*this->output)(index, 0) = wordsAnalysisLevel[i](j, 0);
        }
    }
    this->sentenceCounter++;
}

OpinionInputLayer::~OpinionInputLayer() {
    delete output;
}