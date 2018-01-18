#include "../include/UserInterface.h"
#include "../include/InputParser.h"
#include "../include/FileManager.h"
#include "../include/WordsInputLayer.h"
#include "../include/OpinionInputLayer.h"

#include <Windows.h>
#undef max

//allows to get utf-8 line
std::string readLine()
{
	WCHAR cc[256];
	ULONG n;
	char c[512];
	ReadConsoleW(GetStdHandle(STD_INPUT_HANDLE), cc, RTL_NUMBER_OF(cc), &n, 0);
	WideCharToMultiByte(CP_UTF8, 0, cc, -1, c, 512, 0, 0);
	
	for (int i = 0; i < 512; i++)
		{
		if (c[i] == '\r' || c[i] == '\n')
			c[i] = 0;
	}
	
	return c;
}

std::string UserInterface::rateToWords(float rate) {
    if (rate<-1.0 || rate>1.0)
        return "wystąpił błąd\n";

    else if (rate >= -1.0 && rate<-0.7)
        return "bardzo negatywne\n";

    else if (rate >= -0.7 && rate<-0.4)
        return "negatywne\n";

    else if (rate >= -0.4 && rate<-0.2)
        return "lekko negatywne\n";

    else if (rate >= -0.2 && rate<0.2)
        return "neutralne\n";

    else if (rate >= 0.2 && rate<0.4)
        return "lekko pozytywne\n";

    else if (rate >= 0.4 && rate<0.7)
        return "pozytywne\n";

    else if (rate >= 0.7 && rate<1.0)
        return "bardzo pozytywne\n";

}


void UserInterface::get()
{
    std::cin.ignore(1024, '\n');
    std::cout << "";
    std::cin.get();
}


int UserInterface::chooseMode()
{
    while (1)
    {
        std::string ciag;
        int wybor;
        std::cin >> ciag;
        int i;
        int n = ciag.length();
        for (i = 0; i<n; ++i)
            if (!isdigit(ciag[i])) break;
        if (i == n)
        {
            std::istringstream iss(ciag);
            iss >> wybor;
            return wybor;
        }
        else std::cout << "Niepoprawny wybór. Wybierz cyfrę z przedziału <1,3>\n" << std::endl;
    }
}

void UserInterface::interactiveMode(OpinionAnalysisLevel* opinionAnalysis, WordAnalysisLevel* wordAnalysis)
{
    InputParser parser;

    std::vector<FileManager::sentenceRateStruct> vec;
    string sent;
    float rate;
    string fileName;
    string word;
	Eigen::MatrixXf* result;

	OpinionInputLayer::OpinionInput opinionInput;
	WordsInputLayer::WordsInput wordInput;

    int decision;
    while (1)
    {
        decision = -1;
        cout << "MENU:\n<1>Analiza słowa\n<2>Analiza opinii\n<3>Wyjscie\n";
        decision = chooseMode();
        switch (decision)
        {
            case 1:
                //-----------------------Analyze word------------------------
				word = readLine();
				wordInput.encodedString = parser.encodeString(word);

				result = wordAnalysis->computeOutput(&wordInput);

				cout << rateToWords((*result)(0, 0));
				if ((*result)(1, 0) >= 0.5)
					cout << "neguje\n";
				if ((*result)(2, 0) >= 0.5)
					cout << "wzmacnia\n";
				else if ((*result)(2, 0) <= -0.5)
					cout << "osłabia\n";
                break;
				get();
            case 2:
                //----------------------Analyze opinion-----------------------
                cout << "Wpisz opinię i kliknij Enter." << endl;
                getline(cin, sent);//to ignore new line after option selection
				sent = readLine();

                for (auto sentence : parser.extractSentences(sent))
                {
                    std::vector<Eigen::MatrixXf> analyzedWords;
                    for (auto word : parser.extractWordsFromSentence(sentence))
                    {
						wordInput.encodedString = parser.encodeString(word);
                        analyzedWords.push_back(*wordAnalysis->computeOutput(&wordInput));
                    }

                    opinionInput.sentences.push_back(analyzedWords);
                }

                rate = (*opinionAnalysis->computeOutput(&opinionInput))(0, 0);

				opinionInput.sentences.clear();

                cout << "Nacechowanie wypowiedzi: " + rateToWords(rate) << endl;
                break;
            case 3:
                //----------------------------Exit---------------------------
                cout << "DZIĘKUJĘ ZA SKORZYSTANIE Z PROGRAMU.";
                return;
            default:
                //-----------------------------Error-----------------------------
                cout << "\nBŁĄD WYBORU.\nNacisnij Enter...";
                get();
                break;
        }
    }
}

void UserInterface::activateInteractions(char** argv) {
    //sentence network params file, word network params file
    WordAnalysisLevel* wordNetwork = new WordAnalysisLevel;
    OpinionAnalysisLevel* opinionNetwork = new OpinionAnalysisLevel;

    auto opinionNetworkData = FileManager::readNetworkFile(argv[0],
                                                           { 360, opinionNetwork->NEURONS_1ST_LAYER,
                                                             opinionNetwork->NEURONS_2ND_LAYER,
                                                             opinionNetwork->NEURONS_3RD_LAYER,
                                                             opinionNetwork->NEURONS_OUTPUT_LAYER });
    auto wordNetworkData = FileManager::readNetworkFile(argv[1],
                                                        { 11 * 32, wordNetwork->NEURONS_1ST_LAYER,
                                                          wordNetwork->NEURONS_2ND_LAYER,
                                                          wordNetwork->NEURONS_OUTPUT_LAYER });

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

void UserInterface::performWordLevelTraining(char **argv) {
    //network params file, input file, learning speed, threads, runs
    cout << "Uczenie poziomu analizy słów" << endl;

    WordAnalysisLevel* network = new WordAnalysisLevel;

    auto networkData = FileManager::readNetworkFile(argv[0], { 11 * 32, network->NEURONS_1ST_LAYER, network->NEURONS_2ND_LAYER, network->NEURONS_OUTPUT_LAYER });
    auto trainingExamples = FileManager::readWordTrainingFile(argv[1]);
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
        cout << "Wykonano " << i + 1 << " z " << runs << "... ";
        cost = network->backpropagate(trainingExamples, learningSpeed, threads);
        cout << setprecision(numeric_limits<float>::max_digits10) << cost << endl;

        if (cost <= bestCost)
        {
            cost = bestCost;

            if (i == 0)
                continue;

            networkData = network->getKnownConnections();
        }
    }

    cout << "Zapisywanie..." << endl;
    FileManager::writeNetworkFile(argv[0], networkData);

	for (auto e : trainingExamples)
	{
		delete e.first;
		delete e.second;
	}
}

void UserInterface::performOpinionLevelTraining(char **argv) {
    cout << "Uczenie poziomu analizy opinii" << endl;
    //teach sentence layer
    //opinion network params file, word network params file, input file, learning speed, threads, runs
    WordAnalysisLevel* wordNetwork = new WordAnalysisLevel;
    OpinionAnalysisLevel* opinionNetwork = new OpinionAnalysisLevel;

    auto opinionNetworkData = FileManager::readNetworkFile(argv[0],
                                                           { 360, opinionNetwork->NEURONS_1ST_LAYER, opinionNetwork->NEURONS_2ND_LAYER, opinionNetwork->NEURONS_3RD_LAYER, opinionNetwork->NEURONS_OUTPUT_LAYER });
    auto wordNetworkData = FileManager::readNetworkFile(argv[1],
                                                        { 11 * 32, wordNetwork->NEURONS_1ST_LAYER, wordNetwork->NEURONS_2ND_LAYER, wordNetwork->NEURONS_OUTPUT_LAYER });
    auto trainingExamples_unprocessed = FileManager::readSentenceTrainingFile(argv[2]);
    float learningSpeed = atof(argv[3]);
    int threads = atoi(argv[4]);
    int runs = atoi(argv[5]);

    if (wordNetworkData.size() > 0)
        wordNetwork->initKnownConnections(wordNetworkData);
    if (opinionNetworkData.size() > 0)
        opinionNetwork->initKnownConnections(opinionNetworkData);
    else
        opinionNetwork->initRandomConnections();

    vector < pair<Input*, Eigen::MatrixXf*>> trainingExamples;
    InputParser parser;

    for (auto example : trainingExamples_unprocessed)
    {
        pair<Input*, Eigen::MatrixXf*> prepared;
		OpinionInputLayer::OpinionInput* input = new OpinionInputLayer::OpinionInput;
        auto sentences = parser.extractSentences(example.sentence);
        for (auto sentence : sentences)
        {
            vector<Eigen::MatrixXf> analyzedSentence;
            for (auto word : parser.extractWordsFromSentence(sentence))
            {
				WordsInputLayer::WordsInput wordInput;
				wordInput.encodedString = parser.encodeString(word);
                analyzedSentence.push_back(*wordNetwork->computeOutput(&wordInput));
            }
            input->sentences.push_back(analyzedSentence);
        }
		prepared.first = input;
        prepared.second = new Eigen::MatrixXf(1, 1);
        (*prepared.second)(0, 0) = example.rate1;

        trainingExamples.push_back(prepared);
    }

    float cost = 0, bestCost = numeric_limits<float>::max();

    for (int i = 0; i < runs; i++)
    {
        cout << "Wykonano " << i + 1 << " z " << runs << "... ";

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

    cout << "Zapisywanie..." << endl;
    FileManager::writeNetworkFile(argv[0], opinionNetworkData);

	for (auto e : trainingExamples)
	{
		delete e.first;
		delete e.second;
	}
}
