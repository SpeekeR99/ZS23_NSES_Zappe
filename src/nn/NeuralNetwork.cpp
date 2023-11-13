#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(uint32_t input_size,
                             uint32_t output_size,
                             const std::vector<uint32_t> &hidden_layers_sizes,
                             double learning_rate,
                             uint32_t batch_size,
                             bool softmax_output)
                             : input_size(input_size), output_size(output_size), training_error(0, 1), learning_rate(learning_rate), batch_size(batch_size), gradient(0, 0), softmax_output(softmax_output) {
    this->layers.reserve(hidden_layers_sizes.size() + 2);
    this->layers.emplace_back(std::make_unique<Layer>(this->input_size, act_func_type::linear));
    for (auto &hidden_layer_size : hidden_layers_sizes)
        this->layers.emplace_back(std::make_unique<Layer>(hidden_layer_size));
    this->layers.emplace_back(std::make_unique<Layer>(this->output_size));

    this->init_weights();
    this->reset_gradient();
}

NeuralNetwork::NeuralNetwork(uint32_t input_size,
                             uint32_t output_size,
                             const std::vector<uint32_t> &hidden_layers_sizes,
                             act_func activation_function,
                             double learning_rate,
                             uint32_t batch_size,
                             bool softmax_output)
                             : input_size(input_size), output_size(output_size), training_error(0, 1), learning_rate(learning_rate), batch_size(batch_size), gradient(0, 0), softmax_output(softmax_output) {
    this->layers.reserve(hidden_layers_sizes.size() + 2);
    this->layers.emplace_back(std::make_unique<Layer>(this->input_size, act_func_type::linear));
    for (auto &hidden_layer_size : hidden_layers_sizes)
        this->layers.emplace_back(std::make_unique<Layer>(hidden_layer_size, activation_function));
    this->layers.emplace_back(std::make_unique<Layer>(this->output_size, activation_function));

    this->init_weights();
    this->reset_gradient();
}

NeuralNetwork::NeuralNetwork(uint32_t input_size,
                             uint32_t output_size,
                             const std::vector<uint32_t> &hidden_layers_sizes,
                             act_func_type activation_function,
                             double learning_rate,
                             uint32_t batch_size,
                             bool softmax_output)
                             : input_size(input_size), output_size(output_size), training_error(0, 1), learning_rate(learning_rate), batch_size(batch_size), gradient(0, 0), softmax_output(softmax_output) {
    this->layers.reserve(hidden_layers_sizes.size() + 2);
    this->layers.emplace_back(std::make_unique<Layer>(this->input_size, act_func_type::linear));
    for (auto &hidden_layer_size : hidden_layers_sizes)
        this->layers.emplace_back(std::make_unique<Layer>(hidden_layer_size, activation_function));
    this->layers.emplace_back(std::make_unique<Layer>(this->output_size, activation_function));

    this->init_weights();
    this->reset_gradient();
}

NeuralNetwork::~NeuralNetwork() = default;

void NeuralNetwork::set_input(const Matrix &inputs) {
    auto input = inputs;
    if (input.get_dims()[1] != 1 and input.get_dims()[0] == 1)
        input = input.transpose();
    this->layers[0]->set_inputs(input);
}

Matrix NeuralNetwork::get_output() const {
    if (this->softmax_output)
        return this->layers.back()->get_softmax_output();
    return this->layers.back()->get_output();
}

std::vector<std::unique_ptr<Layer>> &NeuralNetwork::get_layers() {
    return this->layers;
}

void NeuralNetwork::init_weights() {
    for (uint32_t i = 1; i < this->layers.size(); i++) {
        auto &previous_layer = this->layers[i - 1];
        auto &current_layer = this->layers[i];
        current_layer->init_weights(current_layer->get_size(), previous_layer->get_size() + 1); // +1 for bias
    }
}

void NeuralNetwork::reset_gradient() {
    this->gradient = Matrix(0, this->output_size);
}

double NeuralNetwork::loss(const Matrix &expected_output) {
    auto nn_output = this->get_output().transpose();
    if (this->softmax_output) { /* Categorical cross-entropy */
        return -((expected_output * nn_output.log().transpose()).get_value(0, 0));
    } /* Mean squared error */
    auto error = expected_output - nn_output;
    return ((error * error.transpose()).get_value(0, 0)) / 2;
}

void NeuralNetwork::feed_forward() {
    this->layers[0]->activate();
    for (uint32_t i = 1; i < this->layers.size(); i++) {
        auto &previous_layer = this->layers[i - 1];
        auto &current_layer = this->layers[i];

        auto inputs = previous_layer->get_output();
        inputs.add_row({1.}); // bias
        auto weights = current_layer->get_weights();
        inputs = weights * inputs;

        current_layer->set_inputs(inputs);
        current_layer->activate();
    }
}

void NeuralNetwork::back_propagation(const Matrix &expected_output) {
    /* Calculate output layer gradients */
    auto &output_layer = this->layers.back();
    auto output_layer_output = this->get_output(); /* gets softmax output if softmax_output is true */
    auto output_layer_derivative_output = output_layer->get_derivative_output();

    Matrix output_layer_gradients = (expected_output - output_layer_output);
    if (!(this->softmax_output)) { /* Mean squared error */
        for (uint32_t i = 0; i < output_layer_gradients.get_dims()[0]; i++) {
            auto value = output_layer_gradients.get_value(i, 0) * output_layer_derivative_output.get_value(i, 0);
            output_layer_gradients.set_value(i, 0, value);
        }
    }

    this->gradient.add_row(output_layer_gradients.get_values()[0]);

    /* Calculate hidden layers gradients */
    for (uint32_t i = this->layers.size() - 2; i > 0; i--) {
        auto &current_layer = this->layers[i];
        auto current_layer_derivative_output = current_layer->get_derivative_output();
        current_layer_derivative_output.add_row({1.}); // bias

        auto &next_layer = this->layers[i + 1];
        auto next_layer_gradients = this->gradient.get_row(this->gradient.get_dims()[0] - 1);
        auto next_layer_weights = next_layer->get_weights();

        auto current_layer_gradients = next_layer_gradients * next_layer_weights;

        for (uint32_t j = 0; j < current_layer_gradients.get_dims()[0]; j++) {
            auto value = current_layer_gradients.get_value(j, 0) * current_layer_derivative_output.get_value(j, 0);
            current_layer_gradients.set_value(j, 0, value);
        }

        this->gradient.add_row(current_layer_gradients.get_values()[0]);
    }
}

void NeuralNetwork::update_weights() {
    /* TODO: tohle je ze stareho back prop */
//    for (uint32_t i = 1; i < this->layers.size(); i++) {
//        auto &current_layer = this->layers[i];
//        auto current_layer_weights = current_layer->get_weights();
//        auto current_layer_gradients = gradients[this->layers.size() - i - 1];
//
//        auto &previous_layer = this->layers[i - 1];
//        auto previous_layer_output = previous_layer->get_output();
//
//        auto new_weights = Matrix(current_layer->get_size(), previous_layer->get_size() + 1, false);
//        for (uint32_t j = 0; j < current_layer->get_size(); j++) {
//            auto row = current_layer_weights.get_row(j);
//            for (uint32_t k = 0; k < previous_layer->get_size() + 1; k++) {
//                double delta;
//                if (k != previous_layer->get_size())
//                    delta = current_layer_gradients[j] * previous_layer_output[k] * this->learning_rate;
//                else
//                    delta = current_layer_gradients[j] * this->learning_rate;
//
//                new_weights.set_value(j, k, row[k] - delta); // TODO: nevim jestli plus nebo minus
//            }
//        }
//
//        current_layer->set_weights(new_weights);
//    }
}

void NeuralNetwork::train(x_y_matrix &training_data, uint32_t epochs, bool verbose) {
    for (int i = 1; i <= epochs; i++) {
        auto shuffled_indices = std::vector<uint32_t>(training_data.first.get_dims()[0]);
        std::iota(shuffled_indices.begin(), shuffled_indices.end(), 0);
        std::shuffle(shuffled_indices.begin(), shuffled_indices.end(), std::mt19937(std::random_device()()));

        auto batches = std::vector<std::vector<uint32_t>>(training_data.first.get_dims()[0] / this->batch_size,
                                                          std::vector<uint32_t>(this->batch_size));
        for (uint32_t j = 0; j < training_data.first.get_dims()[0] / this->batch_size; j++)
            for (uint32_t k = 0; k < this->batch_size; k++) {
                if (j * this->batch_size + k >= training_data.first.get_dims()[0])
                    break;
                batches[j][k] = shuffled_indices[j * this->batch_size + k];
            }

        double error = 0;
        for (auto &batch : batches) {
            this->reset_gradient();

            auto batch_inputs = Matrix(batch.size(), training_data.first.get_dims()[1], false);
            auto batch_outputs = Matrix(batch.size(), training_data.second.get_dims()[1], false);

            for (uint32_t j = 0; j < batch.size(); j++) {
                auto asdf = training_data.first.get_row(batch[j]);
                batch_inputs.set_row(j, training_data.first.get_row(batch[j]).get_values()[0]);
                batch_outputs.set_row(j, training_data.second.get_row(batch[j]).get_values()[0]);
            }

            for (uint32_t j = 0; j < batch.size(); j++) {
                this->set_input(batch_inputs.get_row(j));
                this->feed_forward();
                error += this->loss(batch_outputs.get_row(j));
                this->back_propagation(batch_outputs.get_row(j));
            }

            this->update_weights();
        }
        error /= training_data.first.get_dims()[0];
        this->training_error.add_row({error});

        if (verbose)
            std::cout << "Epoch: " << i << " Error: " << error << std::endl;

    }
}

Matrix NeuralNetwork::predict(const Matrix &inputs) {
    this->set_input(inputs);
    this->feed_forward();
    return this->get_output();
}
