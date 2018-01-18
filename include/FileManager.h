
#ifndef OPINIONANN_FILEMANAGER_H
#define OPINIONANN_FILEMANAGER_H

#include <Eigen>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <vector>
#include <fstream>

#include "WordsInputLayer.h"
#include "OpinionInputLayer.h"

//class for file i/o operations
class FileManager {
public:
    struct sentenceRateStruct {
        std::string sentence;
        float rate1;
    };
	//reads file with neural network
    static std::vector<std::pair<Eigen::MatrixXf, Eigen::MatrixXf>>
        readNetworkFile(std::string fileName, std::vector<int> neuronsNumbers);
	//writes neural network params to file
    static void writeNetworkFile(std::string fileName, std::vector<std::pair<Eigen::MatrixXf, Eigen::MatrixXf>> network);
	//reads file with word analysis level training data
    static std::vector<std::pair<Input*, Eigen::MatrixXf*>> readWordTrainingFile(std::string fileName);
	//reads file with opinion analysis level training data
    static std::vector<sentenceRateStruct> readSentenceTrainingFile(std::string fileName);
};


#endif //OPINIONANN_FILEMANAGER_H
