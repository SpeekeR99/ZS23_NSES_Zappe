#include "NeuralNetwork.h"

#include <random>

void NeuralNetwork::init_weights() {
    for (uint32_t i = 1; i < this->layers.size(); i++) {
        auto &previous_layer = this->layers[i - 1];
        auto &current_layer = this->layers[i];
        current_layer->init_weights(current_layer->get_size(), previous_layer->get_size() + 1); // +1 for bias
    }
}

NeuralNetwork::NeuralNetwork(uint32_t input_size,
                             uint32_t output_size,
                             const std::vector<uint32_t> &hidden_layers_sizes,
                             double learning_rate)
                             : input_size(input_size), output_size(output_size), learning_rate(learning_rate) {
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
                             act_func activation_function,
                             double learning_rate)
                             : input_size(input_size), output_size(output_size), learning_rate(learning_rate) {
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
                             act_func_type activation_function,
                             double learning_rate)
                             : input_size(input_size), output_size(output_size), learning_rate(learning_rate) {
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
            auto row = weights.get_row(j);
            for (uint32_t k = 0; k < inputs.size(); k++)
                new_input[j] += inputs[k] * row[k];
        }

        current_layer->set_inputs(new_input);
        current_layer->activate();
    }
}

void NeuralNetwork::back_propagation(const std::vector<double> &expected_output) {
    auto gradients = std::vector<std::vector<double>>{};

    /* Calculate overall error */
    this->error = 0.;
    auto predicted_output = this->get_output();
    for (uint32_t i = 0; i < predicted_output.size(); i++) {
        auto delta = expected_output[i] - predicted_output[i];
        this->error += delta * delta;
    }
    this->error /= (2 * predicted_output.size());

    /* Calculate output layer gradients */
    auto &output_layer = this->layers.back();
    auto output_layer_output = output_layer->get_output();
    auto output_layer_derivative_output = output_layer->get_derivative_output();
    auto output_layer_gradients = std::vector<double>(output_layer->get_size(), 0);

    for (uint32_t i = 0; i < output_layer->get_size(); i++)
        output_layer_gradients[i] = (expected_output[i] - output_layer_output[i]) * output_layer_derivative_output[i];

    gradients.emplace_back(output_layer_gradients);

    /* Calculate hidden layers gradients */
    for (uint32_t i = this->layers.size() - 2; i > 0; i--) {

        auto &current_layer = this->layers[i];
        auto current_layer_weights = current_layer->get_weights();
        auto current_layer_derivative_output = current_layer->get_derivative_output();

        auto &next_layer = this->layers[i + 1];
        auto next_layer_gradients = gradients.back();

        auto temp_gradients = std::vector<double>(current_layer->get_size(), 0);
        for (uint32_t j = 0; j < current_layer->get_size(); j++) {
            auto sum = 0.;
            for (uint32_t k = 0; k < next_layer->get_size(); k++)
                sum += next_layer_gradients[k] * current_layer_weights.get_value(k, j);
            temp_gradients[j] = sum * current_layer_derivative_output[j];
        }

        gradients.emplace_back(temp_gradients);
    }

    /* Update weights */
    for (uint32_t i = 1; i < this->layers.size(); i++) {
        auto &current_layer = this->layers[i];
        auto current_layer_weights = current_layer->get_weights();
        auto current_layer_gradients = gradients[this->layers.size() - i - 1];

        auto &previous_layer = this->layers[i - 1];
        auto previous_layer_output = previous_layer->get_output();

        auto new_weights = Matrix(current_layer->get_size(), previous_layer->get_size() + 1, false);
        for (uint32_t j = 0; j < current_layer->get_size(); j++) {
            auto row = current_layer_weights.get_row(j);
            for (uint32_t k = 0; k < previous_layer->get_size() + 1; k++) {
                double delta;
                if (k != previous_layer->get_size())
                    delta = current_layer_gradients[j] * previous_layer_output[k] * this->learning_rate;
                else
                    delta = current_layer_gradients[j] * this->learning_rate;

                new_weights.set_value(j, k, row[k] + delta);
            }
        }

        current_layer->set_weights(new_weights);
    }

}

void NeuralNetwork::train(const std::vector<std::pair<std::vector<double>, std::vector<double>>> &training_data,
                          uint32_t epochs) {
    for (int i = 0; i < epochs; i++) {
        /* shuffle training data */
        auto shuffled_training_data = training_data;
        std::shuffle(shuffled_training_data.begin(), shuffled_training_data.end(), std::mt19937(std::random_device()()));

        for (auto &data : shuffled_training_data) {
            this->set_inputs(data.first);
            this->feed_forward();
            this->back_propagation(data.second);
        }
    }
}

std::vector<double> NeuralNetwork::predict(const std::vector<double> &inputs) {
    this->set_inputs(inputs);
    this->feed_forward();
    return this->get_output();
}
