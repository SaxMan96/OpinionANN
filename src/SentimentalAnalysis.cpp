#include "../include/SentimentalAnalysis.h"


//suggested implementation, may change
//SentimentalAnalysis::SentimentalAnalysis() {
//    this->inputParser = new InputParser();
//    this->wordAnalysisLevel = new WordAnalysisLevel();
//    this->sentenceAnalysisLevel = new SentenceAnalysisLevel();
//    this->opinionAnalysisLevel = new OpinionAnalysisLevel();
//    //here init connections randomly or from file
//}
//
//void SentimentalAnalysis::computeOutput(std::string text) {
//    wordAnalysisResults.clear();
//    sentenceAnalysisResults.clear();
//
//    inputParser->extractSentencesAndSeparators(text);
//    for (int i = 0; i < inputParser->getSentences().size(); i++) {
//        std::vector<std::string> words = InputParser::extractWordsFromSentence(inputParser->getSentences()[i]);
//        for (int j = 0; j < words.size(); j++){
//            std::vector<int> encodedWord = InputParser::encodeString(words[j]);
//            wordAnalysisResults.push_back(*(wordAnalysisLevel->analyzeWord(encodedWord)));
//        }
//        sentenceAnalysisResults.push_back(*(sentenceAnalysisLevel->analyzeSentence(wordAnalysisResults)));
//    }
//    finalResult = (*opinionAnalysisLevel->analyzeOpinion(sentenceAnalysisResults, inputParser->getSeparators()))(0,0);
//}
//
//
//float SentimentalAnalysis::getFinalResult() {
//    return this->finalResult;
//}
//
//SentimentalAnalysis::~SentimentalAnalysis() {
//    delete opinionAnalysisLevel, sentenceAnalysisLevel, wordAnalysisLevel, inputParser;
//}
