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
	//returns string that describes given rate (from -1 - least positive to 1 - most positive)
	static std::string rateToWords(float rate);
	//ignores any unread input
	static void get();
	//allows user to select mode from 2 possibilities and allows to exit application
	// 1 - word analysis
	// 2 - opinion analysis
	// 3 - exit
	static int chooseMode();
	//program loop
	static void interactiveMode(OpinionAnalysisLevel* opinionAnalysis, WordAnalysisLevel* wordAnalysis);
public:
	//prepares and runs interactive mode
    static void activateInteractions(char** argv);
	//trains word analysis
    static void performWordLevelTraining(char** argv);
	//trains opinion level analysis
    static void performOpinionLevelTraining(char** argv);};


#endif //OPINIONANN_USERINTERFACE_H
