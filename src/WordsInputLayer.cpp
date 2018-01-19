#include <iostream>
#include "../include/WordsInputLayer.h"

WordsInputLayer::WordsInputLayer() : InputLayer(this->NEURONS) {
    output = new Eigen::MatrixXf(this->NEURONS, 1); // 1 column, 32 rows for letters in  polish alphabet,
                                            // 11 letters taken into consideration
    output->setZero(11*32, 1);
}


WordsInputLayer::WordsInputLayer(const WordsInputLayer &wordsInputLayer) : InputLayer(this->NEURONS) {
    this->output = new Eigen::MatrixXf(*wordsInputLayer.output);
}

//expected input is a vector of int values. Every int value stands for number of letter - a=0, ą=1, b=2...
// Case insensitive.
void WordsInputLayer::setInput(Input* input) {
	WordsInputLayer::WordsInput* wordsInput = static_cast<WordsInputLayer::WordsInput*>(input);

    output->setZero();
    //first 5 characters coded as 1 of n neurons
    int  i = 0;
    while(i < wordsInput->encodedString.size() && i < 5){
        (*output)((this->ALPHABETH_COUNT*i) + wordsInput->encodedString[i], 0) = 1;
        ++i;
    }

    //last 6
    i = 0;
    while (i < wordsInput->encodedString.size() && i < 6){
        (*output)((this->ALPHABETH_COUNT*(5 + i)) + wordsInput->encodedString[wordsInput->encodedString.size() - 1 - i], 0) = 1;
        ++i;
    }
}

Eigen::MatrixXf* WordsInputLayer::getOutput() {
    return output;
}

WordsInputLayer::~WordsInputLayer() {
    delete output;
}

InputLayer* WordsInputLayer::newClone() {
	return new WordsInputLayer(*this);
}