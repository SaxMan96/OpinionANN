#ifndef OPINIONANN_OPINIONINPUTLAYER_H
#define OPINIONANN_OPINIONINPUTLAYER_H

#include <Eigen>
#include "WordAnalysisLevel.h"

class OpinionInputLayer {
private:
    static const int PARAMETERS_PER_WORD = WordAnalysisLevel::NEURONS_OUTPUT_LAYER;
    static const int MAX_WORDS_CONSIDERED = 24;
    static const int MAX_SENTENCES_CONSIDERED = 5;
    int sentenceCounter = 0;
    Eigen::MatrixXf *output;
public:
    Eigen::MatrixXf* getOutput();
    void addSentence(std::vector<Eigen::MatrixXf> wordsAnalysisResults);
    void reset();
    OpinionInputLayer();
    OpinionInputLayer(const OpinionInputLayer &opinionInputLayer);
    ~OpinionInputLayer();
};

#endif //OPINIONANN_OPINIONINPUTLAYER_H
