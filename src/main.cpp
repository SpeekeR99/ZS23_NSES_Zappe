#include <iostream>
#include "nn/NeuralNetwork.h"
#include "utils/Matrix.h"

int main() {
    std::unique_ptr<NeuralNetwork> nn = std::make_unique<NeuralNetwork>(2, 1, std::vector<uint32_t>{4}, act_func_type::sigmoid, 0.1);

    std::vector<std::pair<std::vector<double>, std::vector<double>>> training_data = {
            {{0, 0}, {0}},
            {{0, 1}, {1}},
            {{1, 0}, {1}},
            {{1, 1}, {0}}
    };

    nn->train(training_data, 10000);

    for (auto &data : training_data) {
        std::cout << "Input: " << data.first[0] << " " << data.first[1] << std::endl;
        std::cout << "Expected output: " << data.second[0] << std::endl;
        std::cout << "Actual output: " << nn->predict(data.first)[0] << std::endl;
        std::cout << std::endl;
    }

    return EXIT_SUCCESS;
}
