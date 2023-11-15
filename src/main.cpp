#include "nn/NeuralNetwork.h"
#include "utils/DataLoader.h"
#include "graphics/Visualization.h"

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

    x_y_pairs data_temp = DataLoader::load_file(
            data_filepath,
            number_of_input_features,
            number_of_output_features,
            ' '
    );
    data_temp = DataLoader::transform_y_to_one_hot(data_temp);
    auto data = DataLoader::split_data(data_temp, 0.8);
    x_y_matrix training_data = DataLoader::transform_to_matrices(data.first);
    x_y_matrix test_data = DataLoader::transform_to_matrices(data.second);

    std::cout << "Training data size: " << training_data.first.get_dims()[0] << std::endl;
    std::cout << "Test data size: " << test_data.first.get_dims()[0] << std::endl;

    nn->train(training_data, 200, true);
    nn->test(test_data);

    Visualization visualization = Visualization();
    visualization.run();

    return EXIT_SUCCESS;
}
