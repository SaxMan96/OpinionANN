#ifndef NEURAL_NETWORK_WORDSINPUTLAYER_H
#define NEURAL_NETWORK_WORDSINPUTLAYER_H


#include <Eigen>
#include <vector>

class WordsInputLayer {
private:
    static const int LETTERS_CONSIDERED = 11;
    static const int ALPHABETH_COUNT = 32;
protected:
    Eigen::MatrixXf *output;
public:
    WordsInputLayer();
    WordsInputLayer(const WordsInputLayer &wordsInputLayer);
    Eigen::MatrixXf* getOutput();
    void computeOutput(std::vector<int> encodedString);
    ~WordsInputLayer();
};


#endif //NEURAL_NETWORK_WORDSINPUTLAYER_H
