#pragma once

#include <vector>
#include <memory>
#include <random>
#include <chrono>
#include "Neuron.h"
#include "../utils/Matrix.h"

enum class act_func_type {
    linear = 0,
    relu,
    sigmoid,
    step,
    sign,
    tanh,
    number_of_activation_functions
};

class Layer {
private:
    uint32_t size;
    std::vector<std::unique_ptr<Neuron>> neurons;
    act_func activation_function;
    act_func derivative_activation_function;
    Matrix weights = Matrix(0, 0);

public:
    explicit Layer(uint32_t size);
    Layer(uint32_t size, act_func activation_function);
    Layer(uint32_t size, act_func_type activation_function);
    ~Layer();

    void activate();

    void init_weights(uint32_t rows, uint32_t cols);
    void set_inputs(const std::vector<double>& inputs);
    void set_activation_function(act_func new_activation_function);
    void set_activation_function(act_func_type new_activation_function);
    void set_derivative_activation_function(act_func new_derivative_activation_function);
    void set_derivative_activation_function(act_func_type new_derivative_activation_function);
    void set_weights(Matrix &new_weights);
    [[nodiscard]] std::vector<double> get_output() const;
    [[nodiscard]] std::vector<double> get_softmax_output() const;
    [[nodiscard]] std::vector<double> get_derivative_output() const;
    [[nodiscard]] uint32_t get_size() const;
    [[nodiscard]] Matrix get_weights() const;
};
