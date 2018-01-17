
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

class FileManager {
public:
    struct sentenceRateStruct {
        std::string sentence;
        float rate1;
    };
    static std::vector<std::pair<Eigen::MatrixXf, Eigen::MatrixXf>>
        readNetworkFile(std::string fileName, std::vector<int> neuronsNumbers);
    static void writeNetworkFile(std::string fileName, std::vector<std::pair<Eigen::MatrixXf, Eigen::MatrixXf>> network);
    static std::vector<std::pair<Input*, Eigen::MatrixXf*>> readWordTrainingFile(std::string fileName);
    static std::vector<sentenceRateStruct> readSentenceTrainingFile(std::string fileName);
};


#endif //OPINIONANN_FILEMANAGER_H
