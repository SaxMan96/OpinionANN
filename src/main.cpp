#include <iostream>
#include <ctime>
#include <vector>
#include <fstream>
#include "../include/WordAnalysisLevel.h"
#include "../include/OpinionInputLayer.h"
#include "../include/OpinionAnalysisLevel.h"
#include "../include/InputParser.h"

using namespace std;

vector<pair<Eigen::MatrixXf, Eigen::MatrixXf>> readNetworkFile(string fileName, vector<int> neuronsNumbers) 
{
	std::vector<pair<Eigen::MatrixXf, Eigen::MatrixXf>> vec;
	string line;
	ifstream myFile(fileName);
	if (myFile.is_open())
	{
		for (int i = 1; i < neuronsNumbers.size(); i++)
		{
			Eigen::MatrixXf weights(neuronsNumbers[i], neuronsNumbers[i - 1]);
			for (int j = 0; j < neuronsNumbers[i]; j++)
				for (int k = 0; k < neuronsNumbers[i - 1]; k++)
					myFile >> weights(j, k);
			
			Eigen::MatrixXf biases(neuronsNumbers[i], 1);
			for (int j = 0; j < neuronsNumbers[i]; j++)
				myFile >> biases(j, 0);

			vec.push_back({ weights, biases });
		}
		myFile.close();
	}
	else cout << "Unable to open file";
	return vec;
}

void writeNetworkFile(string fileName, vector<pair<Eigen::MatrixXf, Eigen::MatrixXf>> network)
{
	string line;
	ofstream myFile(fileName);
	if (myFile.is_open())
	{
		for (auto& layer : network)
			myFile << layer.first << endl << layer.second << endl;
		
		myFile.close();
	}
	else cout << "Unable to open file";
}

vector<pair<vector<int>, Eigen::MatrixXf*>> readWordTrainingFile(string fileName)
{
	std::vector<pair<vector<int>, Eigen::MatrixXf*>> vec;
	string line;

	InputParser parser;

	ifstream myFile(fileName);
	if (myFile.is_open())
	{
		string word;
		int pos = 0;
		while (getline(myFile, line))
		{
			if (line.size() == 0)
				continue;
			Eigen::MatrixXf* expected = new Eigen::MatrixXf(3, 1);

			std::string token = line.substr(0, pos = line.find(' '));
			line.erase(0, pos + 1);
			word = token;
			token = line.substr(0, pos = line.find(' '));
			line.erase(0, pos + 1);
			(*expected)(0, 0) = std::stof(token);
			token = line.substr(0, pos = line.find(' '));
			line.erase(0, pos + 1);
			(*expected)(1, 0) = std::stof(token);
			token = line.substr(0, pos = line.find(' '));
			line.erase(0, pos + 1);
			(*expected)(2, 0) = std::stof(token);

			vec.push_back({ parser.encodeString(line), expected });
		}
		myFile.close();
	}
	else cout << "Unable to open file";
	return vec;
}

int main(int argc, char** argv) {
	std::cout << "Hello, World! I am neural network" << std::endl;

	srand(time(NULL));

	argc--;
	argv++;

	if (argc == 0)
	{
		cout << "Test mode" << endl;
		std::vector<int> input;
		input.push_back(1);
		input.push_back(3);
		input.push_back(16);

		WordAnalysisLevel* wordAnalysisLevel = new WordAnalysisLevel();
		wordAnalysisLevel->initRandomConnections();
		std::cout << *(wordAnalysisLevel->analyzeWord(input)) << std::endl << std::endl;

		std::vector<int> input2;
		input2.push_back(4);
		input2.push_back(8);
		input2.push_back(11);
		input2.push_back(17);

		std::cout << *(wordAnalysisLevel->analyzeWord(input2)) << std::endl << std::endl;

		Eigen::MatrixXf expected(3, 1);
		expected(0, 0) = 0.0f;
		expected(1, 0) = -1.0f;
		expected(2, 0) = 1.0f;

		std::pair<std::vector<int>, Eigen::MatrixXf*> trainingExample(input, &expected);
		std::pair<std::vector<int>, Eigen::MatrixXf*> trainingExample2(input2, &expected);

		for (int i = 0; i < 200; i++)
			std::cout << "WORD Test cost " << i << ": " << wordAnalysisLevel->backpropagate({ trainingExample, trainingExample2 }, 0.05f, 2) << std::endl;

		std::cout << *(wordAnalysisLevel->analyzeWord(input)) << std::endl << std::endl;
		std::cout << *(wordAnalysisLevel->analyzeWord(input2)) << std::endl << std::endl;

		std::cout << std::endl;

		std::vector<Eigen::MatrixXf> wordsAnalysisResult;
		wordsAnalysisResult.push_back(*(wordAnalysisLevel->analyzeWord(input)));
		wordsAnalysisResult.push_back(*(wordAnalysisLevel->analyzeWord(input2)));

		std::vector<Eigen::MatrixXf> wordsAnalysisResult2;
		wordsAnalysisResult2.push_back(*(wordAnalysisLevel->analyzeWord(input2)));
		wordsAnalysisResult2.push_back(*(wordAnalysisLevel->analyzeWord(input)));

		OpinionAnalysisLevel* opinionAnalysisLevel = new OpinionAnalysisLevel();
		opinionAnalysisLevel->initRandomConnections();
		opinionAnalysisLevel->addSentenceToInput(wordsAnalysisResult);
		opinionAnalysisLevel->addSentenceToInput(wordsAnalysisResult2);
		std::cout << *(opinionAnalysisLevel->analyzeOpinion()) << std::endl << std::endl;

		std::vector<std::vector<Eigen::MatrixXf>> trainingSentences;
		trainingSentences.push_back(wordsAnalysisResult);
		trainingSentences.push_back(wordsAnalysisResult2);

		Eigen::MatrixXf expectedSentenceResult(1, 1);
		expectedSentenceResult(0, 0) = 1;

		std::pair<std::vector<std::vector<Eigen::MatrixXf>>, Eigen::MatrixXf*> trainingOpinion(trainingSentences, &expectedSentenceResult);

		for (int i = 0; i < 200; i++)
			std::cout << "OPINION Test cost " << i << ": " << opinionAnalysisLevel->backpropagate({ trainingOpinion }, 0.05f, 1) << std::endl;

		opinionAnalysisLevel->resetInput();
		opinionAnalysisLevel->addSentenceToInput(wordsAnalysisResult);
		opinionAnalysisLevel->addSentenceToInput(wordsAnalysisResult2);
		std::cout << *(opinionAnalysisLevel->analyzeOpinion()) << std::endl << std::endl;

		delete wordAnalysisLevel;
		delete opinionAnalysisLevel;
		getchar();
	}
	else if (argc == 2)
	{
		//interactive mode
		//sentence network params file, word network params file
	}
	else if (argc == 5)
	{
		//teach word layer
		//network params file, input file, learning speed, threads, runs
		cout << "Teaching word layer" << endl;

		WordAnalysisLevel* network = new WordAnalysisLevel;

		auto networkData = readNetworkFile(argv[0], { 11 * 32, network->NEURONS_1ST_LAYER, network->NEURONS_2ND_LAYER, network->NEURONS_OUTPUT_LAYER });
		auto trainingData = readWordTrainingFile(argv[1]);
		float learningSpeed = atof(argv[2]);
		int threads = atoi(argv[3]);
		int runs = atoi(argv[4]);

		float cost, bestCost = numeric_limits<float>::max();
		
		if (networkData.empty())
			network->initRandomConnections();
		else
			network->initKnownConnections(networkData);

		for (int i = 0; i < runs; i++)
		{
			cout << "Run " << i + 1 << " out of " << runs << "... ";
			cost = network->backpropagate(trainingData, learningSpeed, threads);
			cout << cost << endl;

			if (cost < bestCost)
			{
				cost = bestCost;

				if (i == 0)
					continue;

				networkData = network->getKnownConnections();
			}
		}

		cout << "Saving" << endl;
		writeNetworkFile(argv[0], networkData);
	}
	else if (argc == 6)
	{
		cout << "Teaching sentence layer" << endl;
		//teach sentence layer
		//opinion network params file, word network params file, input file, learning speed, threads, runs
		WordAnalysisLevel* wordNetwork = new WordAnalysisLevel;
		OpinionAnalysisLevel* opinionNetwork = new OpinionAnalysisLevel;

		auto opinionNetworkData = readNetworkFile(argv[0], 
		{ wordNetwork->NEURONS_OUTPUT_LAYER, opinionNetwork->NEURONS_1ST_LAYER, opinionNetwork->NEURONS_2ND_LAYER, opinionNetwork->NEURONS_3RD_LAYER, opinionNetwork->NEURONS_OUTPUT_LAYER });
		auto wordNetworkData = readNetworkFile(argv[1], 
		{ 11 * 32, wordNetwork->NEURONS_1ST_LAYER, wordNetwork->NEURONS_2ND_LAYER, wordNetwork->NEURONS_OUTPUT_LAYER });
		float learningSpeed = atof(argv[3]);
		int threads = atoi(argv[4]);
		int runs = atoi(argv[5]);

		float cost = 0, prevCost = numeric_limits<float>::max();

		for (int i = 0; i < runs; i++)
		{
			cout << "Run " << i + 1 << " out of " << runs << "... ";

			cout << cost << endl;
			if (cost < prevCost)
			{
				//save
			}
		}
	}


	return 0;
}