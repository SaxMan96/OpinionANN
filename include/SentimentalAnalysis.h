#ifndef OPINIONANN_SENTIMENTALANALYSIS_H
#define OPINIONANN_SENTIMENTALANALYSIS_H

#include "InputParser.h"
#include "SentenceAnalysisLevel.h"
#include "WordAnalysisLevel.h"
#include "OpinionAnalysisLevel.h"
#include <Eigen>
#include <string>
#include <vector>

class SentimentalAnalysis {
private:
    InputParser* inputParser;
    OpinionAnalysisLevel* opinionAnalysisLevel;
    SentenceAnalysisLevel* sentenceAnalysisLevel;
    WordAnalysisLevel* wordAnalysisLevel;
    std::vector<Eigen::MatrixXf> wordAnalysisResults;
    std::vector<Eigen::MatrixXf> sentenceAnalysisResults;
    float finalResult;
public:
    SentimentalAnalysis();
    void computeOutput(std::string text);
    float getFinalResult();
    ~SentimentalAnalysis();
};

#endif //OPINIONANN_SENTIMENTALANALYSIS_H
