#include <algorithm>
#include <string>
#include "..\include\InputParser.h"

std::vector<std::string> InputParser::extractSentences(std::string text) {
	this->sentences.clear();
	text = toLower(text);
	size_t pos = 0;
	size_t pos1 = 0;
	size_t pos2 = 0;
	size_t pos3 = 0;
	std::string token;
	while ((pos = text.find('\n')) != std::string::npos)
		std::replace(text.begin(), text.end(), '\n', ' ');
	while ((pos1 = text.find('.')) != std::string::npos ||
		(pos2 = text.find('?')) != std::string::npos ||
		(pos3 = text.find('!')) != std::string::npos
		) {
		pos1 = text.find('.');
		pos2 = text.find('?');
		pos3 = text.find('!');
		pos = min(pos1, min(pos2, pos3));
		token = text.substr(0, pos + 1);
		push(token);
		text.erase(0, pos + 1);
	}

	return this->sentences;
}

void InputParser::push(std::string s) {
	while (s.at(0) == ' ')
		s = s.substr(1, s.size());
	if (s == "." || s == "?" || s == "!")
		return;
	sentences.push_back(s);
};

string InputParser::toLower(string text) {
	std::transform(text.begin(), text.end(), text.begin(), ::tolower);
	return text;
}

std::vector<std::string> InputParser::extractWordsFromSentence(std::string sent) {
	sent = toLower(sent);
	std::vector<std::string> ret;
	size_t pos = 0;
	std::string token;
	while (sent.find('\n') != std::string::npos)
		std::replace(sent.begin(), sent.end(), '\n', ' ');
	while ((pos = sent.find(' ')) != std::string::npos) {
		token = sent.substr(0, pos + 1);
		while (token.size()>0 && token.at(0) == ' ')
			token = token.substr(1, token.size());
		if (token.size() == 0) {
			sent.erase(0, pos + 1);
			continue;
		}
		token = checkWordForNonLetters(token);
		if (token.size()>0)
			ret.push_back(token);
		sent.erase(0, pos + 1);
	}
	return ret;
}

string InputParser::checkWordForNonLetters(string word) {
	int pos = 0;
	while (word.size()>0 && !((word.at(pos) <= 'z' && word.at(pos) >= 'a') || (word.at(pos) <= 'Z' && word.at(pos) >= 'A')))
		word.erase(pos, pos + 1);
	pos = word.size() - 1;
	if (word.size() == 0)
		return word;
	char c;
	while (!((word.at(pos) <= 'z' && word.at(pos) >= 'a') || (word.at(pos) <= 'Z' && word.at(pos) >= 'A'))) {
		c = word.at(pos);
		word.erase(pos, pos + 1);
		pos = word.size() - 1;
	}
	return word;
}

std::vector<int> InputParser::encodeString(std::string word) {
	std::vector<int> vec;
	vector<string> letters = { u8"a",u8"ą",u8"b",u8"c",u8"ć",u8"d",u8"e",u8"ę",u8"f",u8"g",u8"h",u8"i",u8"j",u8"k",u8"l",u8"ł",u8"m",u8"n",u8"ń",u8"o",u8"ó",u8"p",u8"r",u8"s",u8"ś",u8"t",u8"u",u8"w",u8"y",u8"z",u8"ź",u8"ż" };
	vector<string> upperCaseLetters = { u8"A",u8"Ą",u8"B",u8"C",u8"Ć",u8"D",u8"E",u8"Ę",u8"F",u8"G",u8"H",u8"I",u8"J",u8"L",u8"L",u8"Ł",u8"M",u8"N",u8"Ń",u8"O",u8"Ó",u8"P",u8"R",u8"S",u8"Ś",u8"T",u8"U",u8"W",u8"Y",u8"Z",u8"Ź",u8"Ż" };
	
	for (int i = 0; i<word.size(); i++)
	{
		int code = -1;
		for (int j = 0; j<letters.size(); j++)
			if (word.substr(i, letters[j].size()) == letters[j]) {
				code = j;
				i += letters[j].size() - 1;
				break;
			}
		for (int j = 0; j<upperCaseLetters.size(); j++)
			if (word.substr(i, upperCaseLetters[j].size()) == upperCaseLetters[j]) {
				code = j;
				i += upperCaseLetters[j].size() - 1;
				break;
			}
		if (code != -1)
			vec.push_back(code);
	}
	return vec;

}