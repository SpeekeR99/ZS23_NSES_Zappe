#pragma once

#include <cmath>

typedef double (*act_func)(double);

class Neuron {
private:
    double input;
    double output;
    double derivative_output;

public:
    explicit Neuron(double input);
    ~Neuron();

    void activate(act_func activation_function, act_func derivative_activation_function);

    void set_input(double new_input);
    [[nodiscard]] double get_input() const;
    [[nodiscard]] double get_output() const;
    [[nodiscard]] double get_derivative_output() const;
};
