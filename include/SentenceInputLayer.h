#ifndef OPINIONANN_SENTENCEINPUTLAYER_H
#define OPINIONANN_SENTENCEINPUTLAYER_H

#include <Eigen>
#include "WordAnalysisLevel.h"

class SentenceInputLayer {
private:
    int PARAMETERS_PER_WORD = WordAnalysisLevel::NEURONS_OUTPUT_LAYER;
    int WORDS_CONSIDERED = 24;
    Eigen::MatrixXf *output;
public:
    Eigen::MatrixXf* getOutput();
    SentenceInputLayer();
    void computeOutput(std::vector<Eigen::MatrixXf> wordAnalysisResults);
    ~SentenceInputLayer();
};

#endif //OPINIONANN_SENTENCEINPUTLAYER_H
