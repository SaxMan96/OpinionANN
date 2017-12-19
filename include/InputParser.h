#ifndef NEURAL_NETWORK_INPUTPARSER_H
#define NEURAL_NETWORK_INPUTPARSER_H

#include <vector>
#include <string>
#include <Eigen>

class InputParser {
private:
    std::vector<std::string> sentences; // every part of text separated from others by '.', '!' or '?'
public:
    void extractSentences(std::string text); //save results in class fields
    static std::vector<std::string> extractWordsFromSentence(std::string sentence);
    static std::vector<int> encodeString(std::string word);
    std::vector<std::string> getSentences();
    std::vector<int> getSeparators();
};


#endif //NEURAL_NETWORK_INPUTPARSER_H
