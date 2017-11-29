//
// Created by Delebrith on 24.11.2017.
//

#ifndef NEURAL_NETWORK_WORDSINPUTLAYER_H
#define NEURAL_NETWORK_WORDSINPUTLAYER_H


#include <Eigen>
#include <vector>

class WordsInputLayer {
private:
    Eigen::MatrixXf *output;
public:
    WordsInputLayer();
    Eigen::MatrixXf* getOutput();
    void calculateOutput(std::vector<int> encodedString);
    ~WordsInputLayer();
};


#endif //NEURAL_NETWORK_WORDSINPUTLAYER_H
