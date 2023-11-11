cmake_minimum_required(VERSION 3.24)
project(ZS23_NSES_Zappe)

set(CMAKE_CXX_STANDARD 23)

add_executable(
        ZS23_NSES_Zappe
        src/main.cpp
        src/nn/Neuron.cpp
        src/nn/Neuron.h
        src/nn/Layer.cpp
        src/nn/Layer.h
        src/nn/NeuralNetwork.cpp src/nn/NeuralNetwork.h)
