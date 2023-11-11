#pragma once

#include <iostream>
#include "Layer.h"

class NeuralNetwork {
private:
    uint32_t input_size;
    uint32_t output_size;
    std::vector<std::unique_ptr<Layer>> layers;

public:
    NeuralNetwork(uint32_t input_size,
                  uint32_t output_size,
                  const std::vector<uint32_t> &hidden_layers_sizes);
    NeuralNetwork(uint32_t input_size,
                  uint32_t output_size,
                  const std::vector<uint32_t> &hidden_layers_sizes,
                  act_func activation_function);
    NeuralNetwork(uint32_t input_size,
                  uint32_t output_size,
                  const std::vector<uint32_t> &hidden_layers_sizes,
                  act_func_type activation_function);
    ~NeuralNetwork();

    void set_inputs(const std::vector<double> &inputs);
    [[nodiscard]] std::vector<double> get_output() const;

    void forward_propagation();
};
