#pragma once

#include <cmath>

typedef double (*act_func)(double);
enum class act_func_type {
    linear = 0,
    relu,
    sigmoid,
    step,
    sign,
    tanh,
    number_of_activation_functions
};

class Neuron {
private:
    double input;
    double output;
    double derivative_output;
    act_func activation_function;
    act_func derivative_activation_function;

public:
    explicit Neuron(double input);
    Neuron(double input, act_func activation_function);
    Neuron(double input, act_func_type activation_function);
    ~Neuron();

    void activate();

    void set_input(double new_input);
    void set_activation_function(act_func new_activation_function);
    void set_activation_function(act_func_type new_activation_function);
    [[nodiscard]] double get_output() const;
    [[nodiscard]] double get_derivative_output() const;
};
