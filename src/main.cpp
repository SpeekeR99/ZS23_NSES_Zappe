#include <iostream>
#include "nn/NeuralNetwork.h"
#include "utils/Matrix.h"
#include "utils/DataLoader.h"

int main() {
    std::unique_ptr<NeuralNetwork> nn = std::make_unique<NeuralNetwork>(2, 5, std::vector<uint32_t>{8}, act_func_type::sigmoid, 0.1, true);

    x_y_pairs training_data = DataLoader::load_file("data/tren_data1___23.txt", 2, 1, ' ');
    training_data = DataLoader::transform_y_to_one_hot(training_data);

    nn->train(training_data, 1000, true);

    for (auto &data : training_data) {
        std::cout << "Input: " << data.first[0] << " " << data.first[1] << std::endl;
        auto temp_expected = std::max_element(data.second.begin(), data.second.end()) - data.second.begin();
        std::cout << "Expected output: " << temp_expected << std::endl;
        auto temp_predicted = nn->predict(data.first);
        auto argmax = std::max_element(temp_predicted.begin(), temp_predicted.end()) - temp_predicted.begin();
        std::cout << "Actual output: " << argmax << std::endl;
        std::cout << std::endl;
    }

    return EXIT_SUCCESS;
}
