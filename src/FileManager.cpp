#include <fstream>
#include "../include/FileManager.h"
#include "../include/InputParser.h"


std::vector<std::pair<Eigen::MatrixXf, Eigen::MatrixXf>> FileManager::readNetworkFile
        (std::string fileName, std::vector<int> neuronsNumbers)
{
    std::vector<std::pair<Eigen::MatrixXf, Eigen::MatrixXf>> vec;
    std::string line;
    std::ifstream myFile(fileName);
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
    else std::cout << "Unable to open file";
    return vec;
}

void writeNetworkFile(std::string fileName, std::vector<std::pair<Eigen::MatrixXf, Eigen::MatrixXf>> network)
{
    std::string line;
    std::ofstream myFile(fileName);
    if (myFile.is_open())
    {
        for (auto& layer : network)
            myFile << std::setprecision(std::numeric_limits<float>::max_digits10) << layer.first << std::endl
                   << layer.second << std::endl;

        myFile.close();
    }
    else std::cout << "Unable to open file";
}

std::vector<std::pair<std::vector<int>, Eigen::MatrixXf*>> FileManager::readWordTrainingFile(std::string fileName)
{
    std::vector<std::pair<std::vector<int>, Eigen::MatrixXf*>> vec;
        std::string line;

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

std::vector<FileManager::sentenceRateStruct> FileManager::readSentenceTrainingFile(std::string fileName) {
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