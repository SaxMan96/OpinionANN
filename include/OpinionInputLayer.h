#ifndef OPINIONANN_OPINIONINPUTLAYER_H
#define OPINIONANN_OPINIONINPUTLAYER_H

#include <Eigen>
#include "SentenceAnalysisLevel.h"

class OpinionInputLayer {
private:
    static const int PARAMETERS_PER_SENTENCE = SentenceAnalysisLevel::NEURONS_OUTPUT_LAYER;
    static const int MAX_SENTENCES_CONSIDERED = 60;
    static const int SEPARATORS_POSSIBLE = 10; // separators are '.' ',' '!' '?' '(' ')' ':' ';' '-' and '...' .
                                                // They're encoded as 1 of n in matrix
    Eigen::MatrixXf *output;
public:
    Eigen::MatrixXf* getOutput();
    void computeOutput(std::vector<Eigen::MatrixXf> sentenceAnalysisResults, std::vector<int> encodedSeparators);
    OpinionInputLayer();
    ~OpinionInputLayer();
};

#endif //OPINIONANN_OPINIONINPUTLAYER_H
