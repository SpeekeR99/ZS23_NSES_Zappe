#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(uint32_t input_size,
                             uint32_t output_size,
                             const std::vector<uint32_t> &hidden_layers_sizes)
                             : input_size(input_size), output_size(output_size){
    this->layers.reserve(hidden_layers_sizes.size() + 2);
    this->layers.emplace_back(std::make_unique<Layer>(this->input_size));
    for (auto &hidden_layer_size : hidden_layers_sizes)
        this->layers.emplace_back(std::make_unique<Layer>(hidden_layer_size));
    this->layers.emplace_back(std::make_unique<Layer>(this->output_size));
}

NeuralNetwork::NeuralNetwork(uint32_t input_size,
                             uint32_t output_size,
                             const std::vector<uint32_t> &hidden_layers_sizes,
                             act_func activation_function)
                             : input_size(input_size), output_size(output_size){
    this->layers.reserve(hidden_layers_sizes.size() + 2);
    this->layers.emplace_back(std::make_unique<Layer>(this->input_size, activation_function));
    for (auto &hidden_layer_size : hidden_layers_sizes)
        this->layers.emplace_back(std::make_unique<Layer>(hidden_layer_size, activation_function));
    this->layers.emplace_back(std::make_unique<Layer>(this->output_size, activation_function));
}

NeuralNetwork::NeuralNetwork(uint32_t input_size,
                             uint32_t output_size,
                             const std::vector<uint32_t> &hidden_layers_sizes,
                             act_func_type activation_function)
                             : input_size(input_size), output_size(output_size){
    this->layers.reserve(hidden_layers_sizes.size() + 2);
    this->layers.emplace_back(std::make_unique<Layer>(this->input_size, activation_function));
    for (auto &hidden_layer_size : hidden_layers_sizes)
        this->layers.emplace_back(std::make_unique<Layer>(hidden_layer_size, activation_function));
    this->layers.emplace_back(std::make_unique<Layer>(this->output_size, activation_function));
}

NeuralNetwork::~NeuralNetwork() = default;

void NeuralNetwork::set_inputs(const std::vector<double> &inputs) {
    this->layers[0]->set_inputs(inputs);
}

std::vector<double> NeuralNetwork::get_output() const {
    return this->layers.back()->get_output();
}

void NeuralNetwork::forward_propagation() {
    this->layers[0]->activate();
    for (uint32_t i = 1; i < this->layers.size(); i++) {
        auto old_outputs = this->layers[i - 1]->get_output();
        auto new_input = this->layers[i]->get_bias();
        for (auto &old_output: old_outputs)
            new_input += old_output;
        auto input_vector = std::vector<double>{};
        for (auto j = 0; j < this->layers[j]->get_size(); j++)
            input_vector.push_back(new_input);
        this->layers[i]->set_inputs(input_vector);
        this->layers[i]->activate();
    }
}
