#ifndef OPINIONANN_USERINTERFACE_H
#define OPINIONANN_USERINTERFACE_H

#include <string>
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <sstream>
#include "OpinionAnalysisLevel.h"
#include "WordAnalysisLevel.h"

class UserInterface {
public:
    static std::string rateToWords(float rate);
    static void get();
    static bool isFloat(std::string string);
    static int chooseMode();
    static void interactiveMode(OpinionAnalysisLevel* opinionAnalysis, WordAnalysisLevel* wordAnalysis);
    static void activateInteractions(char** argv);
    static void performWordLevelTraining(char** argv);
    static void performOpinionLevelTraining(char** argv);};


#endif //OPINIONANN_USERINTERFACE_H
