#pragma once

#include <iostream>
#include <algorithm>
#include "Layer.h"
#include "../utils/Matrix.h"
#include "../utils/DataLoader.h"

class NeuralNetwork {
private:
    uint32_t input_size;
    uint32_t output_size;
    std::vector<std::unique_ptr<Layer>> layers;
    Matrix training_error;
    double learning_rate;
    Matrix gradient;
    uint32_t batch_size;
    bool softmax_output;

    void init_weights();
    void reset_gradient();
    void set_input(const Matrix &inputs);
    [[nodiscard]] Matrix get_output() const;
    void back_propagation(const Matrix &expected_output);

public:
    void feed_forward();
    NeuralNetwork(uint32_t input_size,
                  uint32_t output_size,
                  const std::vector<uint32_t> &hidden_layers_sizes,
                  double learning_rate = 0.1,
                  uint32_t batch_size = 1,
                  bool softmax_output = false);
    NeuralNetwork(uint32_t input_size,
                  uint32_t output_size,
                  const std::vector<uint32_t> &hidden_layers_sizes,
                  act_func activation_function,
                  double learning_rate = 0.1,
                  uint32_t batch_size = 1,
                  bool softmax_output = false);
    NeuralNetwork(uint32_t input_size,
                  uint32_t output_size,
                  const std::vector<uint32_t> &hidden_layers_sizes,
                  act_func_type activation_function,
                  double learning_rate = 0.1,
                  uint32_t batch_size = 1,
                  bool softmax_output = false);
    ~NeuralNetwork();

    [[nodiscard]] std::vector<std::unique_ptr<Layer>> &get_layers();

    void train(x_y_matrix &training_data, uint32_t epochs, bool verbose = false);
    Matrix predict(const Matrix &inputs);
};
