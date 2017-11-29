cmake_minimum_required(VERSION 3.8)
project(OpinionANN)

set(CMAKE_CXX_STANDARD 11)

#location of Eigen header files
#location of this project header files
include_directories(
	eigen/Eigen
	headers)

set(SOURCE_FILES
	src/main.cpp
	src/InputParser.cpp
	src/OpinionAnalysisNetwork.cpp
	src/SentenceAnalysisNetwork.cpp
	src/WordAnalysisLevel.cpp
        src/WordsInputLayer.cpp
	src/WordsLayer.cpp)
add_executable(OpinionANN ${SOURCE_FILES})