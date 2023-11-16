#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(uint32_t input_size,
                             uint32_t output_size,
                             const std::vector<uint32_t> &hidden_layers_sizes,
                             double learning_rate,
                             uint32_t batch_size,
                             bool softmax_output)
                             : input_size(input_size), output_size(output_size), training_error(0, 1), learning_rate(learning_rate), batch_size(batch_size), gradient{}, softmax_output(softmax_output) {
    this->layers.reserve(hidden_layers_sizes.size() + 2); /* +2 for input and output layers */
    this->layers.emplace_back(std::make_shared<Layer>(this->input_size, act_func_type::linear)); /* input layer is linear */
    for (auto &hidden_layer_size : hidden_layers_sizes)
        this->layers.emplace_back(std::make_shared<Layer>(hidden_layer_size));
    this->layers.emplace_back(std::make_shared<Layer>(this->output_size));

    /* Initialize weights */
    this->init_weights();
    /* Initialize gradient */
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
    this->layers.reserve(hidden_layers_sizes.size() + 2); /* +2 for input and output layers */
    this->layers.emplace_back(std::make_shared<Layer>(this->input_size, act_func_type::linear)); /* input layer is linear */
    for (auto &hidden_layer_size : hidden_layers_sizes)
        this->layers.emplace_back(std::make_shared<Layer>(hidden_layer_size, activation_function));
    this->layers.emplace_back(std::make_shared<Layer>(this->output_size, activation_function));

    /* Initialize weights */
    this->init_weights();
    /* Initialize gradient */
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
    this->layers.reserve(hidden_layers_sizes.size() + 2); /* +2 for input and output layers */
    this->layers.emplace_back(std::make_shared<Layer>(this->input_size, act_func_type::linear)); /* input layer is linear */
    for (auto &hidden_layer_size : hidden_layers_sizes)
        this->layers.emplace_back(std::make_shared<Layer>(hidden_layer_size, activation_function));
    this->layers.emplace_back(std::make_shared<Layer>(this->output_size, activation_function));

    /* Initialize weights */
    this->init_weights();
    /* Initialize gradient */
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

Matrix NeuralNetwork::get_training_error() const {
    return this->training_error;
}

void NeuralNetwork::init_weights() {
    for (uint32_t i = 1; i < this->layers.size(); i++) {
        auto &previous_layer = this->layers[i - 1];
        auto &current_layer = this->layers[i];
        current_layer->init_weights(current_layer->get_size(), previous_layer->get_size() + 1); // +1 for bias
    }
}

void NeuralNetwork::reset_gradient() {
    this->gradient.clear(); /* Clear previous gradient */
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
    this->layers[0]->activate(); /* input layer activation (linear, so just copy inputs) */
    for (uint32_t i = 1; i < this->layers.size(); i++) {
        auto &previous_layer = this->layers[i - 1];
        auto &current_layer = this->layers[i];

        auto inputs = previous_layer->get_output(); /* previous layer output */
        inputs.add_row({1.}); /* add bias */

        auto weights = current_layer->get_weights();

        inputs = weights * inputs; /* matrix multiplication */

        current_layer->set_inputs(inputs);
        current_layer->activate();
    }
}

void NeuralNetwork::back_propagation(const Matrix &expected_output) {
    std::vector<Matrix> cached_gradients = {};  /* Cache gradients for hidden layers */

    /* Calculate output layer gradients */
    auto &output_layer = this->layers.back();
    auto output_layer_output = this->get_output(); /* gets softmax output if softmax_output is true */
    auto output_layer_derivative_output = output_layer->get_derivative_output();

    auto &previous_layer = this->layers[this->layers.size() - 2];
    auto previous_layer_output = previous_layer->get_output();
    previous_layer_output.add_row({1.}); /* bias */

    Matrix output_layer_gradients = (expected_output - output_layer_output.transpose());
    if (!(this->softmax_output)) { /* Mean squared error */
        for (uint32_t i = 0; i < output_layer_gradients.get_dims()[1]; i++) {
            auto value = output_layer_gradients.get_value(0, i) * output_layer_derivative_output.get_value(i, 0);
            output_layer_gradients.set_value(0, i, value);
        }
    }

    cached_gradients.emplace_back(output_layer_gradients); /* Cache this part of the gradient for previous layer */
    output_layer_gradients = output_layer_gradients.transpose() * previous_layer_output.transpose();

    this->gradient[this->gradient.size() - 1].emplace_back(output_layer_gradients);

    /* Calculate hidden layers gradients */
    for (uint32_t i = this->layers.size() - 2; i > 0; i--) {
        auto &current_layer = this->layers[i];
        auto current_layer_derivative_output = current_layer->get_derivative_output();
        current_layer_derivative_output.add_row({1.}); /* bias */

        auto &next_layer = this->layers[i + 1];
        auto next_layer_gradients = cached_gradients[this->layers.size() - 2 - i];
        auto next_layer_weights = next_layer->get_weights();
        next_layer_weights.remove_col(next_layer_weights.get_dims()[1] - 1); /* remove bias */

        auto &previous_layer = this->layers[i - 1];
        auto previous_layer_output = previous_layer->get_output();
        previous_layer_output.add_row({1.}); /* bias */

        auto current_layer_gradients = (next_layer_gradients * next_layer_weights).transpose();
        for (uint32_t j = 0; j < current_layer_gradients.get_dims()[1]; j++) {
            auto value = current_layer_gradients.get_value(0, j) * current_layer_derivative_output.get_value(j, 0);
            current_layer_gradients.set_value(0, j, value);
        }

        cached_gradients.emplace_back(current_layer_gradients.transpose()); /* Cache this part of the gradient for previous layer */
        current_layer_gradients = current_layer_gradients * previous_layer_output.transpose();

        this->gradient[i - 1].emplace_back(current_layer_gradients);
    }
}

void NeuralNetwork::update_weights() {
    /* Average gradients over batches */
    std::vector<Matrix> averaged_gradients = {};

    for (auto &grad_matrices : this->gradient) {
        /* Sum gradients */
        Matrix averaged_gradient(grad_matrices[0].get_dims()[0], grad_matrices[0].get_dims()[1], false);
        for (auto &grad : grad_matrices) {
            for (uint32_t i = 0; i < grad.get_dims()[0]; i++)
                for (uint32_t j = 0; j < grad.get_dims()[1]; j++)
                    averaged_gradient.set_value(i, j, averaged_gradient.get_value(i, j) + grad.get_value(i, j));
        }
        /* Average gradients */
        for (uint32_t i = 0; i < averaged_gradient.get_dims()[0]; i++)
            for (uint32_t j = 0; j < averaged_gradient.get_dims()[1]; j++)
                averaged_gradient.set_value(i, j, averaged_gradient.get_value(i, j) / grad_matrices.size());

        averaged_gradients.emplace_back(std::move(averaged_gradient));
    }

    for (uint32_t i = 1; i < this->layers.size(); i++) {
        auto &current_layer = this->layers[i];

        auto weights = current_layer->get_weights();
        auto gradients = averaged_gradients[i - 1];

        /* Update weights */
        auto new_weights = weights + (gradients * this->learning_rate);
        current_layer->set_weights(new_weights);
    }
}

void NeuralNetwork::train(x_y_matrix &training_data, uint32_t epochs, bool verbose) {
    for (int i = 1; i <= epochs; i++)
        this->train_one_step(training_data, i, verbose);
}

void NeuralNetwork::train_one_step(x_y_matrix &training_data, uint32_t epoch, bool verbose) {
    /* Shuffle training data */
    auto shuffled_indices = std::vector<uint32_t>(training_data.first.get_dims()[0]);
    std::iota(shuffled_indices.begin(), shuffled_indices.end(), 0);
    std::shuffle(shuffled_indices.begin(), shuffled_indices.end(), std::mt19937(std::random_device()()));

    /* Create batches */
    auto batches = std::vector<std::vector<uint32_t>>(training_data.first.get_dims()[0] / this->batch_size,
                                                      std::vector<uint32_t>(this->batch_size));
    for (uint32_t j = 0; j < training_data.first.get_dims()[0] / this->batch_size; j++)
        for (uint32_t k = 0; k < this->batch_size; k++) {
            if (j * this->batch_size + k >= training_data.first.get_dims()[0])
                break;
            batches[j][k] = shuffled_indices[j * this->batch_size + k];
        }

    /* Train on batches */
    double error = 0; /* Average error over all batches */
    for (auto &batch : batches) { /* For each batch */
        this->reset_gradient(); /* Reset gradient */

        /* Prepare batch inputs and outputs */
        auto batch_inputs = Matrix(batch.size(), training_data.first.get_dims()[1], false);
        auto batch_outputs = Matrix(batch.size(), training_data.second.get_dims()[1], false);

        for (uint32_t j = 0; j < batch.size(); j++) {
            batch_inputs.set_row(j, training_data.first.get_row(batch[j]).get_values()[0]);
            batch_outputs.set_row(j, training_data.second.get_row(batch[j]).get_values()[0]);
        }

        /* Train on batch */
        for (uint32_t j = 0; j < batch.size(); j++) {
            this->set_input(batch_inputs.get_row(j)); /* Set input */
            this->feed_forward(); /* Feed forward */
            error += this->loss(batch_outputs.get_row(j)); /* Calculate error */
            this->back_propagation(batch_outputs.get_row(j)); /* Back propagation */
        }

        /* Update weights */
        this->update_weights();
    }
    /* Calculate average error over all batches */
    error /= training_data.first.get_dims()[0];
    this->training_error.add_row({error});

    if (verbose) /* Print epoch and error */
        std::cout << "Epoch: " << epoch << " Error: " << error << std::endl;
}

double NeuralNetwork::test(x_y_matrix &test_data) {
    /* Calculate accuracy */
    double correct = 0;

    for (uint32_t i = 0; i < test_data.first.get_dims()[0]; i++) {
        auto predicted_output = this->predict(test_data.first.get_row(i));
        auto expected_output = test_data.second.get_row(i);
        auto predicted_output_max = predicted_output.argmax();
        auto expected_output_max = expected_output.argmax();

        if (predicted_output_max == expected_output_max)
                correct++; /* Correct prediction */
    }

    /* Return accuracy as correct predictions over total predictions */
    return correct / test_data.first.get_dims()[0];
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
