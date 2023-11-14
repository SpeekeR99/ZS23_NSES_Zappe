#include <iostream>
#include "nn/NeuralNetwork.h"
#include "utils/Matrix.h"
#include "utils/DataLoader.h"

int main() {
    std::unique_ptr<NeuralNetwork> nn = std::make_unique<NeuralNetwork>(2, 5, std::vector<uint32_t>{8}, act_func_type::sigmoid, 0.001, 50, true);

    x_y_pairs training_data_temp = DataLoader::load_file("data/tren_data1___23.txt", 2, 1, ' ');
    training_data_temp = DataLoader::transform_y_to_one_hot(training_data_temp);
    x_y_matrix training_data = DataLoader::transform_to_matrices(training_data_temp);

    nn->train(training_data, 10000, true);

    for (auto &pair : training_data_temp) {
        Matrix input = Matrix(1, 2, {pair.first});
        Matrix output = Matrix(1, 5, {pair.second});
        Matrix prediction = nn->predict(input);
        std::cout << "Input: " << input << std::endl;
        std::cout << "Expected output: " << output << std::endl;
        std::cout << "Prediction: " << prediction.transpose() << std::endl;
        std::cout << std::endl;
    }

    return EXIT_SUCCESS;
}
