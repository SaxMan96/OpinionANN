#include <iostream>
#include <iomanip>
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
			myFile << setprecision(numeric_limits<float>::max_digits10) << layer.first << endl << layer.second << endl;
		
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

			vec.push_back({ parser.encodeString(word), expected });
		}
		myFile.close();
	}
	else cout << "Unable to open file";
	return vec;
}

struct sentenceRateStruct {
	std::string sentence;
	float rate1;
};
vector<sentenceRateStruct> readSentenceTrainingFile(string fileName) {
	std::vector<sentenceRateStruct> vec;
	string line;
	string token;
	ifstream myFile(fileName);
	if (myFile.is_open())
	{
		string sent;

		float rate;
		int pos = 0;
		while (getline(myFile, line))
		{
			rate = std::stof(line.substr(0, pos = line.find(' ')));
			line.erase(0, pos + 1);
			sent = line.substr(0, pos = line.find('\n'));
			line.erase(0, pos + 1);
			sentenceRateStruct temp = { sent,rate };
			vec.push_back(temp);
		}
		myFile.close();
	}
	else cout << "Unable to open file";
	return vec;
}

string rateToWords(float rate) {
	if (rate<-1.0 || rate>1.0)
		return "wystąpił błąd\n";

	else if (rate >= -1.0 && rate<-0.7)
		return "bardzo negatywne\n";

	else if (rate >= -0.7 && rate<-0.4)
		return "negatywne\n";

	else if (rate >= -0.4 && rate<-0.1)
		return "lekkonegatywne\n";

	else if (rate >= -0.1 && rate<0.1)
		return "neutralne\n";

	else if (rate >= 0.1 && rate<0.4)
		return "lekko pozytywne\n";

	else if (rate >= 0.4 && rate<0.7)
		return "pozytywne\n";

	else if (rate >= 0.7 && rate<1.0)
		return "bardzo pozytywne\n";

}
void clrscr()
{
	cin.sync();
	for (int i = 0; i<25; ++i)
		cout << "\n";
}

void get()
{
	cin.ignore(1024, '\n');
	cout << "";
	cin.get();
}
bool isFloat(string myString) {
	std::istringstream iss(myString);
	float f;
	iss >> noskipws >> f;
	return iss.eof() && !iss.fail();
}

int wyborMenu()
{
	while (1)
	{
		string ciag;
		int wybor;
		cin >> ciag;
		int i;
		int n = ciag.length();
		for (i = 0; i<n; ++i)
			if (!isdigit(ciag[i])) break;
		if (i == n)
		{
			istringstream iss(ciag);
			iss >> wybor;
			return wybor;
		}
		else cout << "Niepoprawny wybor. Wybierz cyfre z przedzialu <1,3>\n" << endl;
	}
}

void interactiveMode(OpinionAnalysisLevel* opinionAnalyzis, WordAnalysisLevel* wordAnalyzis)
{
	InputParser parser;

	std::vector<sentenceRateStruct> vec;
	string sent;
	float rate;
	string fileName;
	string word;

	int decyzja;
	while (1)
	{
		decyzja = -1;
		cout << "MENU:\n<1>Analiza słowa\n<2>Analiza opini\n<3>Wyjscie\n";
		decyzja = wyborMenu();
		switch (decyzja)
		{
		case 1:
			//-----------------------Wczytaj z pliku------------------------
			clrscr();
			cout << "Podaj słowo i naciśnij ENTER" << endl;
			cin >> word;
			cout << (*wordAnalyzis->analyzeWord(parser.encodeString(word)));
			get();
			clrscr();
			break;
		case 2:
			//----------------------Wczytaj z klawiatury-----------------------
			clrscr();
			cout << "Wpisz opinię i kliknij Enter." << endl;
			getline(cin, sent);//to ignore new line after option selection
			getline(cin, sent);

			for (auto sentence : parser.extractSentences(sent))
			{
				std::vector<Eigen::MatrixXf> analyzedWords;
				for (auto word : parser.extractWordsFromSentence(sentence))
				{
					analyzedWords.push_back(*wordAnalyzis->analyzeWord(parser.encodeString(word)));
				}

				opinionAnalyzis->addSentenceToInput(analyzedWords);
			}

			rate = (*opinionAnalyzis->analyzeOpinion())(0, 0);

			opinionAnalyzis->resetInput();

			cout << "Nacechowanie zdania: " + rateToWords(rate) << endl;
			break;
		case 3:
			//----------------------------Wyjscie---------------------------
			clrscr();
			cout << "DZIEKUJE ZA SKORZYSTANIE Z PROGRAMU.";
			return;
		default:
			//-----------------------------Blad-----------------------------
			cout << "\nBLAD WYBORU.\nNacisnij Enter...";
			get();
			clrscr();
			break;
		}
	}
}

int main(int argc, char** argv) 
{
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
		WordAnalysisLevel* wordNetwork = new WordAnalysisLevel;
		OpinionAnalysisLevel* opinionNetwork = new OpinionAnalysisLevel;

		auto opinionNetworkData = readNetworkFile(argv[0],
		{ 360, opinionNetwork->NEURONS_1ST_LAYER, opinionNetwork->NEURONS_2ND_LAYER, opinionNetwork->NEURONS_3RD_LAYER, opinionNetwork->NEURONS_OUTPUT_LAYER });
		auto wordNetworkData = readNetworkFile(argv[1],
		{ 11 * 32, wordNetwork->NEURONS_1ST_LAYER, wordNetwork->NEURONS_2ND_LAYER, wordNetwork->NEURONS_OUTPUT_LAYER });

		if (wordNetworkData.empty())
			wordNetwork->initRandomConnections();
		else
			wordNetwork->initKnownConnections(wordNetworkData);
		if (opinionNetworkData.empty())
			wordNetwork->initRandomConnections();
		else
			opinionNetwork->initKnownConnections(opinionNetworkData);

		interactiveMode(opinionNetwork, wordNetwork);

		delete wordNetwork;
		delete opinionNetwork;
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
			cout << setprecision(numeric_limits<float>::max_digits10) << cost << endl;

			if (cost <= bestCost)
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
		cout << "Teaching opinion layer" << endl;
		//teach sentence layer
		//opinion network params file, word network params file, input file, learning speed, threads, runs
		WordAnalysisLevel* wordNetwork = new WordAnalysisLevel;
		OpinionAnalysisLevel* opinionNetwork = new OpinionAnalysisLevel;

		auto opinionNetworkData = readNetworkFile(argv[0], 
		{ 360, opinionNetwork->NEURONS_1ST_LAYER, opinionNetwork->NEURONS_2ND_LAYER, opinionNetwork->NEURONS_3RD_LAYER, opinionNetwork->NEURONS_OUTPUT_LAYER });
		auto wordNetworkData = readNetworkFile(argv[1], 
		{ 11 * 32, wordNetwork->NEURONS_1ST_LAYER, wordNetwork->NEURONS_2ND_LAYER, wordNetwork->NEURONS_OUTPUT_LAYER });
		auto trainingExamples_unprocessed = readSentenceTrainingFile(argv[2]);
		float learningSpeed = atof(argv[3]);
		int threads = atoi(argv[4]);
		int runs = atoi(argv[5]);

		if (wordNetworkData.size() > 0)
			wordNetwork->initKnownConnections(wordNetworkData);
		if (opinionNetworkData.size() > 0)
			opinionNetwork->initKnownConnections(opinionNetworkData);
		else
			opinionNetwork->initRandomConnections();

		vector < pair<vector<vector<Eigen::MatrixXf>>, Eigen::MatrixXf*>> trainingExamples;
		InputParser parser;

		for (auto example : trainingExamples_unprocessed)
		{
			pair<vector<vector<Eigen::MatrixXf>>, Eigen::MatrixXf*> prepared;

			auto sentences = parser.extractSentences(example.sentence);
			for (auto sentence : sentences)
			{
				vector<Eigen::MatrixXf> analyzedSentence;
				for (auto word : parser.extractWordsFromSentence(sentence))
				{
					analyzedSentence.push_back(*wordNetwork->analyzeWord(parser.encodeString(word)));
				}

				prepared.first.push_back(analyzedSentence);
			}

			prepared.second = new Eigen::MatrixXf(1, 1);
			(*prepared.second)(0, 0) = example.rate1;

			trainingExamples.push_back(prepared);
		}

		float cost = 0, bestCost = numeric_limits<float>::max();

		for (int i = 0; i < runs; i++)
		{
			cout << "Run " << i + 1 << " out of " << runs << "... ";

			cost = opinionNetwork->backpropagate(trainingExamples, learningSpeed, threads);

			cout << setprecision(numeric_limits<float>::max_digits10) << cost << endl;
			if (cost < bestCost)
			{
				cost = bestCost;

				if (i == 0)
					continue;

				opinionNetworkData = opinionNetwork->getKnownConnections();
			}
		}

		cout << "Saving" << endl;
		writeNetworkFile(argv[0], opinionNetworkData);
	}


	return 0;
}