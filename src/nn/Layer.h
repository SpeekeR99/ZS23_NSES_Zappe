#pragma once

#include <vector>
#include <memory>
#include "Neuron.h"

class Layer {
private:
    uint32_t size;
    std::vector<std::unique_ptr<Neuron>> neurons;
    double bias = 1.0;

public:
    explicit Layer(uint32_t size);
    Layer(uint32_t size, act_func activation_function);
    Layer(uint32_t size, act_func_type activation_function);
    ~Layer();

    void activate();

    void set_inputs(const std::vector<double>& inputs);
    void set_activation_function(act_func new_activation_function);
    void set_activation_function(act_func_type new_activation_function);
    [[nodiscard]] std::vector<double> get_output() const;
    [[nodiscard]] double get_bias() const;
    [[nodiscard]] uint32_t get_size() const;
};
