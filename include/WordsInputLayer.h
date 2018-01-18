#ifndef NEURAL_NETWORK_WORDSINPUTLAYER_H
#define NEURAL_NETWORK_WORDSINPUTLAYER_H


#include <Eigen>
#include <vector>
#include "Input.h"
#include "InputLayer.h"

class WordsInputLayer : public InputLayer {
private:
    static const int LETTERS_CONSIDERED = 11;
    static const int ALPHABETH_COUNT = 32;

	static const int NEURONS = LETTERS_CONSIDERED * ALPHABETH_COUNT;
protected:
	Eigen::MatrixXf *output;
public:
	struct WordsInput : Input
	{
		//word encoded as numbers, one for each consecutive letter
		std::vector<int> encodedString;
	};

	Eigen::MatrixXf* getOutput();
	void setInput(Input* input);

	InputLayer* newClone();

    WordsInputLayer();
    WordsInputLayer(const WordsInputLayer &wordsInputLayer);
    ~WordsInputLayer();
};


#endif //NEURAL_NETWORK_WORDSINPUTLAYER_H
