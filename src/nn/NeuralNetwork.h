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
    std::vector<std::shared_ptr<Layer>> layers;
    Matrix training_error;
    double learning_rate;
    std::vector<std::vector<Matrix>> gradient;
    uint32_t batch_size;
    bool softmax_output;

    void set_input(const Matrix &inputs);
    [[nodiscard]] Matrix get_output() const;

    void init_weights();
    void feed_forward();
    void reset_gradient();
    double loss(const Matrix &expected_output);
    void back_propagation(const Matrix &expected_output);
    void update_weights();

public:
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

    [[nodiscard]] const std::vector<std::shared_ptr<Layer>> &get_layers() const;

    void train(x_y_matrix &training_data, uint32_t epochs, bool verbose = false);
    void test(x_y_matrix &test_data);
    Matrix predict(const Matrix &inputs);

    NeuralNetwork &operator=(const NeuralNetwork &nn);
    friend std::ostream &operator<<(std::ostream &os, const NeuralNetwork &nn);
};
