//
// Created by MateuszDorobek on 07.01.2018.
//

#ifndef PSZTUTILITIES_INPUTPARSER_H
#define PSZTUTILITIES_INPUTPARSER_H
#include <vector>
#include <string>
#include <iostream>
using namespace std;
class InputParser {
private:
	std::vector<std::string> sentences; // every part of text separated from others by '.', '!' or '?'
public:
	std::vector<std::string> extractSentences(std::string text); //save results in class fields
	static std::vector<std::string> extractWordsFromSentence(std::string sentence);
	static std::vector<int> encodeString(std::string word);
	std::vector<std::vector<int>> encodeSentence();
	std::vector<std::string> getSentences() { return sentences; };
private:
	void push(std::string);
	static string checkWordForNonLetters(string basic_string);
	static string toLower(string text);
};


#endif //NEURAL_NETWORK_INPUTPARSER_H

