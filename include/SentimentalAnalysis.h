#ifndef OPINIONANN_SENTIMENTALANALYSIS_H
#define OPINIONANN_SENTIMENTALANALYSIS_H

#include "InputParser.h"
#include "WordAnalysisLevel.h"
#include "OpinionAnalysisLevel.h"
#include <Eigen>
#include <string>
#include <vector>

class SentimentalAnalysis {
private:
    InputParser* inputParser;
    OpinionAnalysisLevel* opinionAnalysisLevel;
    WordAnalysisLevel* wordAnalysisLevel;
    float finalResult;
public:
    SentimentalAnalysis();
    void computeOutput(std::string text);
    float getFinalResult();
    ~SentimentalAnalysis();
};

#endif //OPINIONANN_SENTIMENTALANALYSIS_H
