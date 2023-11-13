#include <iostream>
#include "nn/NeuralNetwork.h"
#include "utils/Matrix.h"
#include "utils/DataLoader.h"

int main() {
    std::unique_ptr<NeuralNetwork> nn = std::make_unique<NeuralNetwork>(2, 2, std::vector<uint32_t>{4}, act_func_type::sigmoid, 0.1, 2, true);

    x_y_pairs training_data_temp = DataLoader::load_file("data/xor.txt", 2, 1, ' ');
    training_data_temp = DataLoader::transform_y_to_one_hot(training_data_temp);
    x_y_matrix training_data = DataLoader::transform_to_matrices(training_data_temp);

    nn->train(training_data, 10, true);

    return EXIT_SUCCESS;
}
