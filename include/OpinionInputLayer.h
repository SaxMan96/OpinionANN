#ifndef OPINIONANN_OPINIONINPUTLAYER_H
#define OPINIONANN_OPINIONINPUTLAYER_H

#include <Eigen>
#include "InputLayer.h"
#include "Input.h"

class OpinionInputLayer : public InputLayer {
private:
    static const int PARAMETERS_PER_WORD = 3;
    static const int MAX_WORDS_CONSIDERED = 24; //within a single sentence
	static const int MAX_SENTENCES_CONSIDERED = 5;

	static const int NEURONS = PARAMETERS_PER_WORD * MAX_WORDS_CONSIDERED * MAX_SENTENCES_CONSIDERED;

    int sentenceCounter = 0;
    Eigen::MatrixXf *output;

	//adds single sentence to input
	void addSentence(std::vector<Eigen::MatrixXf> wordsAnalysisResults);
	//resets input
	void reset();

public:
	struct OpinionInput : ::Input
	{
		//vector of sentences, each sentence is a vector of analysed words
		std::vector<std::vector<Eigen::MatrixXf>> sentences;
	};
	
    Eigen::MatrixXf* getOutput();
	void setInput(Input* input);

	InputLayer* newClone();

    OpinionInputLayer();
    OpinionInputLayer(const OpinionInputLayer &opinionInputLayer);
    ~OpinionInputLayer();
};

#endif //OPINIONANN_OPINIONINPUTLAYER_H
