#include <iostream>
#include "nn/NeuralNetwork.h"
#include "utils/Matrix.h"
#include "utils/DataLoader.h"

const std::string data_filepath = "data/tren_data1___23.txt";
const uint32_t number_of_input_features = 2;
const uint32_t number_of_output_features = 1;
const uint32_t number_of_classes = 5;

int main() {
    std::unique_ptr<NeuralNetwork> nn = std::make_unique<NeuralNetwork>(
            number_of_input_features,
            number_of_classes,
            std::vector<uint32_t>{8},
            act_func_type::relu,
            0.01,
            10,
            true
    );

    x_y_pairs training_data_temp = DataLoader::load_file(
            data_filepath,
            number_of_input_features,
            number_of_output_features,
            ' '
    );
    training_data_temp = DataLoader::transform_y_to_one_hot(training_data_temp);
    x_y_matrix training_data = DataLoader::transform_to_matrices(training_data_temp);

    nn->train(training_data, 1000, true);

    for (auto &pair : training_data_temp) {
        Matrix input = Matrix(1, number_of_input_features, {pair.first});
        Matrix output = Matrix(1, number_of_classes, {pair.second});
        Matrix prediction = nn->predict(input);
        std::cout << "Input: " << input << std::endl;
        std::cout << "Expected output: " << output << std::endl;
        std::cout << "Prediction: " << prediction.transpose() << std::endl;
        std::cout << std::endl;
    }

    return EXIT_SUCCESS;
}
