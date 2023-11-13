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
    double error = 0.;
    double learning_rate;
    bool softmax_output = false;

    void init_weights();
    void back_propagation(const Matrix &expected_output);

public:
    void feed_forward();
    NeuralNetwork(uint32_t input_size,
                  uint32_t output_size,
                  const std::vector<uint32_t> &hidden_layers_sizes,
                  double learning_rate = 0.1,
                  bool softmax_output = false);
    NeuralNetwork(uint32_t input_size,
                  uint32_t output_size,
                  const std::vector<uint32_t> &hidden_layers_sizes,
                  act_func activation_function,
                  double learning_rate = 0.1,
                  bool softmax_output = false);
    NeuralNetwork(uint32_t input_size,
                  uint32_t output_size,
                  const std::vector<uint32_t> &hidden_layers_sizes,
                  act_func_type activation_function,
                  double learning_rate = 0.1,
                  bool softmax_output = false);
    ~NeuralNetwork();

    void set_inputs(const Matrix &inputs);
    [[nodiscard]] Matrix get_output() const;
    [[nodiscard]] std::vector<std::unique_ptr<Layer>> &get_layers();

    void train(x_y_matrix &training_data, uint32_t epochs, bool verbose = false);
    Matrix predict(const Matrix &inputs);
};
