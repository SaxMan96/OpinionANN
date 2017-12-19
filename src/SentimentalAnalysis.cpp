#include "../include/SentimentalAnalysis.h"

//TODO implement InputParser methods and test

//suggested implementation, may change
//SentimentalAnalysis::SentimentalAnalysis() {
//    this->inputParser = new InputParser();
//    this->wordAnalysisLevel = new WordAnalysisLevel();
//    this->opinionAnalysisLevel = new OpinionAnalysisLevel();
//    //here init connections randomly or from file
//}
//
//void SentimentalAnalysis::computeOutput(std::string text) {
//
//    std::vector<Eigen::MatrixXf> wordsAnalysisResults;
//    inputParser->extractSentences(text);
//    for (int i = 0; i < inputParser->getSentences().size(); i++) {
//        wordsAnalysisResults.clear();
//        std::vector<std::string> words = InputParser::extractWordsFromSentence(inputParser->getSentences()[i]);
//        for (int j = 0; j < words.size(); j++){
//            std::vector<int> encodedWord = InputParser::encodeString(words[j]);
//            wordsAnalysisResults.push_back(*(wordAnalysisLevel->analyzeWord(encodedWord)));
//        }
//        opinionAnalysisLevel->addSentenceToInput(wordsAnalysisResults);
//    }
//    finalResult = (*opinionAnalysisLevel->analyzeOpinion())(0,0);
//}
//
//
//float SentimentalAnalysis::getFinalResult() {
//    return this->finalResult;
//}
//
//SentimentalAnalysis::~SentimentalAnalysis() {
//    delete opinionAnalysisLevel, wordAnalysisLevel, inputParser;
//}
