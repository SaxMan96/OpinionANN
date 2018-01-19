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
	//divides string by characters '.', '!', '?'
	std::vector<std::string> extractSentences(std::string text); //save results in class fields
	//divides given sentence into words (by whitespace)
	static std::vector<std::string> extractWordsFromSentence(std::string sentence);
	//encodes word
	//each letter from polish alphabet is encoded as value from 0 (for letter 'a'),
	//non-letter characters are omited
	static std::vector<int> encodeString(std::string word);
	//returns last opinion parsed into sentences
	std::vector<std::string> getSentences() { return sentences; };
private:
	//adds given string to sentences,
	//trimming begin and ignoring if first non-whitespace character is '.', '!' or '?'
	void push(std::string);
};


#endif //NEURAL_NETWORK_INPUTPARSER_H

