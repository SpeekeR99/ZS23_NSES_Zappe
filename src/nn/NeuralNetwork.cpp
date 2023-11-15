#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(uint32_t input_size,
                             uint32_t output_size,
                             const std::vector<uint32_t> &hidden_layers_sizes,
                             double learning_rate,
                             uint32_t batch_size,
                             bool softmax_output)
                             : input_size(input_size), output_size(output_size), training_error(0, 1), learning_rate(learning_rate), batch_size(batch_size), gradient{}, softmax_output(softmax_output) {
    this->layers.reserve(hidden_layers_sizes.size() + 2);
    this->layers.emplace_back(std::make_shared<Layer>(this->input_size, act_func_type::linear));
    for (auto &hidden_layer_size : hidden_layers_sizes)
        this->layers.emplace_back(std::make_shared<Layer>(hidden_layer_size));
    this->layers.emplace_back(std::make_shared<Layer>(this->output_size));

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
                             : input_size(input_size), output_size(output_size), training_error(0, 1), learning_rate(learning_rate), batch_size(batch_size), gradient{}, softmax_output(softmax_output) {
    this->layers.reserve(hidden_layers_sizes.size() + 2);
    this->layers.emplace_back(std::make_shared<Layer>(this->input_size, act_func_type::linear));
    for (auto &hidden_layer_size : hidden_layers_sizes)
        this->layers.emplace_back(std::make_shared<Layer>(hidden_layer_size, activation_function));
    this->layers.emplace_back(std::make_shared<Layer>(this->output_size, activation_function));

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
                             : input_size(input_size), output_size(output_size), training_error(0, 1), learning_rate(learning_rate), batch_size(batch_size), gradient{}, softmax_output(softmax_output) {
    this->layers.reserve(hidden_layers_sizes.size() + 2);
    this->layers.emplace_back(std::make_shared<Layer>(this->input_size, act_func_type::linear));
    for (auto &hidden_layer_size : hidden_layers_sizes)
        this->layers.emplace_back(std::make_shared<Layer>(hidden_layer_size, activation_function));
    this->layers.emplace_back(std::make_shared<Layer>(this->output_size, activation_function));

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

const std::vector<std::shared_ptr<Layer>> &NeuralNetwork::get_layers() const {
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
    this->gradient.clear();
    this->gradient.reserve(this->layers.size() - 1);
    for (uint32_t i = 1; i < this->layers.size(); i++)
        this->gradient.emplace_back();
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
    std::vector<Matrix> cached_gradients = {};

    /* Calculate output layer gradients */
    auto &output_layer = this->layers.back();
    auto output_layer_output = this->get_output(); /* gets softmax output if softmax_output is true */
    auto output_layer_derivative_output = output_layer->get_derivative_output();

    auto &previous_layer = this->layers[this->layers.size() - 2];
    auto previous_layer_output = previous_layer->get_output();
    previous_layer_output.add_row({1.}); // bias

    Matrix output_layer_gradients = (expected_output - output_layer_output.transpose());
    if (!(this->softmax_output)) { /* Mean squared error */
        for (uint32_t i = 0; i < output_layer_gradients.get_dims()[1]; i++) {
            auto value = output_layer_gradients.get_value(0, i) * output_layer_derivative_output.get_value(i, 0);
            output_layer_gradients.set_value(0, i, value);
        }
    }

    cached_gradients.emplace_back(output_layer_gradients);
    output_layer_gradients = output_layer_gradients.transpose() * previous_layer_output.transpose();

    this->gradient[this->gradient.size() - 1].emplace_back(output_layer_gradients);

    /* Calculate hidden layers gradients */
    for (uint32_t i = this->layers.size() - 2; i > 0; i--) {
        auto &current_layer = this->layers[i];
        auto current_layer_derivative_output = current_layer->get_derivative_output();
        current_layer_derivative_output.add_row({1.}); // bias

        auto &next_layer = this->layers[i + 1];
        auto next_layer_gradients = cached_gradients[this->layers.size() - 2 - i];
        auto next_layer_weights = next_layer->get_weights();
        next_layer_weights.remove_col(next_layer_weights.get_dims()[1] - 1); // remove bias

        auto &previous_layer = this->layers[i - 1];
        auto previous_layer_output = previous_layer->get_output();
        previous_layer_output.add_row({1.}); // bias

        auto current_layer_gradients = (next_layer_gradients * next_layer_weights).transpose();
        for (uint32_t j = 0; j < current_layer_gradients.get_dims()[1]; j++) {
            auto value = current_layer_gradients.get_value(0, j) * current_layer_derivative_output.get_value(j, 0);
            current_layer_gradients.set_value(0, j, value);
        }

        cached_gradients.emplace_back(current_layer_gradients.transpose());
        current_layer_gradients = current_layer_gradients * previous_layer_output.transpose();

        this->gradient[i - 1].emplace_back(current_layer_gradients);
    }
}

void NeuralNetwork::update_weights() {
    std::vector<Matrix> averaged_gradients = {};

    for (auto &grad_matrices : this->gradient) {
        Matrix averaged_gradient(grad_matrices[0].get_dims()[0], grad_matrices[0].get_dims()[1], false);
        for (auto &grad : grad_matrices) {
            for (uint32_t i = 0; i < grad.get_dims()[0]; i++)
                for (uint32_t j = 0; j < grad.get_dims()[1]; j++)
                    averaged_gradient.set_value(i, j, averaged_gradient.get_value(i, j) + grad.get_value(i, j));
        }
        for (uint32_t i = 0; i < averaged_gradient.get_dims()[0]; i++)
            for (uint32_t j = 0; j < averaged_gradient.get_dims()[1]; j++)
                averaged_gradient.set_value(i, j, averaged_gradient.get_value(i, j) / grad_matrices.size());
        averaged_gradients.emplace_back(std::move(averaged_gradient));
    }

    for (uint32_t i = 1; i < this->layers.size(); i++) {
        auto &current_layer = this->layers[i];

        auto weights = current_layer->get_weights();
        auto gradients = averaged_gradients[i - 1];

        auto new_weights = weights + (gradients * this->learning_rate);
        current_layer->set_weights(new_weights);
    }
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

        if (verbose && !(i % (epochs / 100)))
            std::cout << "Epoch: " << i << " Error: " << error << std::endl;

    }
}

void NeuralNetwork::test(x_y_matrix &test_data) {
    double correct = 0;
    for (uint32_t i = 0; i < test_data.first.get_dims()[0]; i++) {
        this->set_input(test_data.first.get_row(i));
        this->feed_forward();
        auto nn_output = this->get_output();
        auto expected_output = test_data.second.get_row(i);
        auto nn_output_max = nn_output.argmax();
        auto expected_output_max = expected_output.argmax();
        if (nn_output_max == expected_output_max)
                correct++;
    }
    std::cout << "Accuracy: " << correct / test_data.first.get_dims()[0] * 100 << " %" << std::endl;
}

Matrix NeuralNetwork::predict(const Matrix &inputs) {
    this->set_input(inputs);
    this->feed_forward();
    return this->get_output();
}

NeuralNetwork &NeuralNetwork::operator=(const NeuralNetwork &nn) {
    this->input_size = nn.input_size;
    this->output_size = nn.output_size;
    this->layers.clear();
    this->layers.reserve(nn.layers.size());
    this->layers = nn.layers;
    this->init_weights();
    this->training_error = nn.training_error;
    this->learning_rate = nn.learning_rate;
    this->batch_size = nn.batch_size;
    this->reset_gradient();
    this->softmax_output = nn.softmax_output;
    return *this;
}

std::ostream &operator<<(std::ostream &os, const NeuralNetwork &nn) {
    os << "NeuralNetwork: " << std::endl;
    os << "    Input size: " << nn.input_size << std::endl;
    os << "    Output size: " << nn.output_size << std::endl;
    os << "    Hidden Layers: " << std::endl;
    for (int i = 1; i < nn.layers.size() - 1; i++)
        os << "        Hidden layer " << i << " size: " << nn.layers[i]->get_size() << std::endl;
    os << "    Learning rate: " << nn.learning_rate << std::endl;
    os << "    Batch size: " << nn.batch_size << std::endl;
    os << "    Softmax output: " << nn.softmax_output << std::endl;
    return os;
}
