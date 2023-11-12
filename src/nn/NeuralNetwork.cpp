#include "NeuralNetwork.h"

void NeuralNetwork::init_weights() {
    for (uint32_t i = 1; i < this->layers.size(); i++) {
        auto &previous_layer = this->layers[i - 1];
        auto &current_layer = this->layers[i];
        current_layer->init_weights(current_layer->get_size(), previous_layer->get_size() + 1); // +1 for bias
    }
}

NeuralNetwork::NeuralNetwork(uint32_t input_size,
                             uint32_t output_size,
                             const std::vector<uint32_t> &hidden_layers_sizes)
                             : input_size(input_size), output_size(output_size){
    this->layers.reserve(hidden_layers_sizes.size() + 2);
    this->layers.emplace_back(std::make_unique<Layer>(this->input_size, act_func_type::linear));
    for (auto &hidden_layer_size : hidden_layers_sizes)
        this->layers.emplace_back(std::make_unique<Layer>(hidden_layer_size));
    this->layers.emplace_back(std::make_unique<Layer>(this->output_size));

    this->init_weights();
}

NeuralNetwork::NeuralNetwork(uint32_t input_size,
                             uint32_t output_size,
                             const std::vector<uint32_t> &hidden_layers_sizes,
                             act_func activation_function)
                             : input_size(input_size), output_size(output_size){
    this->layers.reserve(hidden_layers_sizes.size() + 2);
    this->layers.emplace_back(std::make_unique<Layer>(this->input_size, act_func_type::linear));
    for (auto &hidden_layer_size : hidden_layers_sizes)
        this->layers.emplace_back(std::make_unique<Layer>(hidden_layer_size, activation_function));
    this->layers.emplace_back(std::make_unique<Layer>(this->output_size, activation_function));

    this->init_weights();
}

NeuralNetwork::NeuralNetwork(uint32_t input_size,
                             uint32_t output_size,
                             const std::vector<uint32_t> &hidden_layers_sizes,
                             act_func_type activation_function)
                             : input_size(input_size), output_size(output_size){
    this->layers.reserve(hidden_layers_sizes.size() + 2);
    this->layers.emplace_back(std::make_unique<Layer>(this->input_size, act_func_type::linear));
    for (auto &hidden_layer_size : hidden_layers_sizes)
        this->layers.emplace_back(std::make_unique<Layer>(hidden_layer_size, activation_function));
    this->layers.emplace_back(std::make_unique<Layer>(this->output_size, activation_function));

    this->init_weights();
}

NeuralNetwork::~NeuralNetwork() = default;

void NeuralNetwork::set_inputs(const std::vector<double> &inputs) {
    this->layers[0]->set_inputs(inputs);
}

std::vector<double> NeuralNetwork::get_output() const {
    return this->layers.back()->get_output();
}

std::vector<std::unique_ptr<Layer>> &NeuralNetwork::get_layers() {
    return this->layers;
}

void NeuralNetwork::feed_forward() {
    this->layers[0]->activate();
    for (uint32_t i = 1; i < this->layers.size(); i++) {
        auto &previous_layer = this->layers[i - 1];
        auto &current_layer = this->layers[i];

        auto inputs = previous_layer->get_output();
        inputs.emplace_back(1); // bias
        auto weights = current_layer->get_weights();

        auto new_input = std::vector<double>(current_layer->get_size(), 0);
        for (uint32_t j = 0; j < current_layer->get_size(); j++) {
            for (uint32_t k = 0; k < inputs.size(); k++)
                new_input[j] += inputs[k] * weights.get_value(j, k);
        }

        current_layer->set_inputs(new_input);
        current_layer->activate();
    }
}
