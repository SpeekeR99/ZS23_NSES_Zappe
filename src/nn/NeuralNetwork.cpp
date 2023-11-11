#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(uint32_t input_size,
                             uint32_t output_size,
                             const std::vector<uint32_t> &hidden_layers_sizes)
                             : input_size(input_size), output_size(output_size) {
    this->layers.reserve(hidden_layers_sizes.size() + 2);
    this->layers.emplace_back(std::make_unique<Layer>(input_size));
    for (auto &hidden_layer_size : hidden_layers_sizes)
        this->layers.emplace_back(std::make_unique<Layer>(hidden_layer_size));
    this->layers.emplace_back(std::make_unique<Layer>(output_size));
}

NeuralNetwork::NeuralNetwork(uint32_t input_size,
                             uint32_t output_size,
                             const std::vector<uint32_t> &hidden_layers_sizes,
                             act_func activation_function)
                             : input_size(input_size), output_size(output_size) {
    this->layers.reserve(hidden_layers_sizes.size() + 2);
    this->layers.emplace_back(std::make_unique<Layer>(input_size, activation_function));
    for (auto &hidden_layer_size : hidden_layers_sizes)
        this->layers.emplace_back(std::make_unique<Layer>(hidden_layer_size, activation_function));
    this->layers.emplace_back(std::make_unique<Layer>(output_size, activation_function));
}

NeuralNetwork::NeuralNetwork(uint32_t input_size,
                             uint32_t output_size,
                             const std::vector<uint32_t> &hidden_layers_sizes,
                             act_func_type activation_function)
                             : input_size(input_size), output_size(output_size) {
    this->layers.reserve(hidden_layers_sizes.size() + 2);
    this->layers.emplace_back(std::make_unique<Layer>(input_size, activation_function));
    for (auto &hidden_layer_size : hidden_layers_sizes)
        this->layers.emplace_back(std::make_unique<Layer>(hidden_layer_size, activation_function));
    this->layers.emplace_back(std::make_unique<Layer>(output_size, activation_function));
}

NeuralNetwork::~NeuralNetwork() = default;
