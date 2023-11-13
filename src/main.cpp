#include <iostream>
#include "nn/NeuralNetwork.h"
#include "utils/Matrix.h"
#include "utils/DataLoader.h"

int main() {
    std::unique_ptr<NeuralNetwork> nn = std::make_unique<NeuralNetwork>(2, 2, std::vector<uint32_t>{5}, act_func_type::sigmoid, 0.1);

    x_y_pairs training_data_temp = DataLoader::load_file("data/tren_data1___23.txt", 2, 1, ' ');
    training_data_temp = DataLoader::transform_y_to_one_hot(training_data_temp);
    x_y_matrix training_data = DataLoader::transform_to_matrices(training_data_temp);

    Matrix input(1, 2, {{1, 2}});
    input = input.transpose();
    nn->set_inputs(input);
    nn->feed_forward();
    Matrix output = nn->get_output();
    std::cout << output << std::endl;

//    nn->train(training_data, 1000, true);

    return EXIT_SUCCESS;
}
