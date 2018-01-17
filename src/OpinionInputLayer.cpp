#include <Eigen>
#include "../include/OpinionInputLayer.h"

OpinionInputLayer::OpinionInputLayer() : InputLayer(this->NEURONS) {
    this->output = new Eigen::MatrixXf(this->NEURONS, 1);
    this->output->setZero();
}

OpinionInputLayer::OpinionInputLayer(const OpinionInputLayer &opinionInputLayer) : InputLayer(this->NEURONS) {
    this->output = new Eigen::MatrixXf(*opinionInputLayer.output);
}

Eigen::MatrixXf* OpinionInputLayer::getOutput() {
    return this->output;
}
/*
 * Function places results of calculations from word analysis level in opinion analysis level's input layer.
 * Consecutive sentences start every 72 neurons and fill as much of them as they need,
 * depending on how many words they include.
 */
void OpinionInputLayer::addSentence(std::vector<Eigen::MatrixXf> wordsAnalysisLevel){
    if (this->sentenceCounter >= this->MAX_SENTENCES_CONSIDERED) return;
    for (int i = 0; i < this->MAX_WORDS_CONSIDERED && i < wordsAnalysisLevel.size(); i++){
        for (int j = 0; j < this->PARAMETERS_PER_WORD; j++){
            int index = 
				(this->sentenceCounter * this->MAX_WORDS_CONSIDERED * this->PARAMETERS_PER_WORD) + // * neurons per sentence
				(i*this->PARAMETERS_PER_WORD) + // * neurons per word
				j;
            (*this->output)(index, 0) = wordsAnalysisLevel[i](j, 0);
        }
    }
    this->sentenceCounter++;
}

void OpinionInputLayer::reset() {
    this->sentenceCounter = 0;
    this->output->setZero();
}

void OpinionInputLayer::setInput(::Input* input)
{
	OpinionInput* opinionInput = static_cast<OpinionInput*>(input);

	reset();

	for (auto& sentene : opinionInput->sentences)
		addSentence(sentene);
}

OpinionInputLayer::~OpinionInputLayer() {
    delete output;
}

InputLayer* OpinionInputLayer::newClone() {
	return new OpinionInputLayer(*this);
}