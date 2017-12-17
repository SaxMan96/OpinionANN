#include <iostream>
#include "../include/WordsInputLayer.h"

WordsInputLayer::WordsInputLayer(){
    output = new Eigen::MatrixXf(this->ALPHABETH_COUNT * this->LETTERS_CONSIDERED, 1); // 1 column, 32 rows for letters in  polish alphabet,
                                            // 11 letters taken into consideration
    output->setZero(11*32, 1);
}


//expected input is a vector of int values. Every int value stands for number of letter - a=1, b=2, c=3...
// Case insensitive.
void WordsInputLayer::computeOutput(std::vector<int> encodedString) {
    output->setZero();
    //first 5 characters coded as 1 of n neurons
    int  i = 0;
    while(i < encodedString.size() && i < 5){
        (*output)((32*i) + encodedString[i], 0) = 1;
        ++i;
    }

    //last 6
    i = 0;
    while (i < encodedString.size() && i < 6){
        (*output)((32*(5 + i)) + encodedString[encodedString.size() - 1 - i], 0) = 1;
        ++i;
    }
}

Eigen::MatrixXf* WordsInputLayer::getOutput() {
    return output;
}

WordsInputLayer::~WordsInputLayer() {
    delete output;
}