#pragma once

#include <cmath>

/** Activation function */
typedef double (*act_func)(double);

/**
 * Class representing a neuron
 * My neuron can only store an input and if the layer gives him a function to activate with, it can activate and
 * store the output and derivative output
 */
class Neuron {
private:
    /** Input of the neuron (already weighted) */
    double input;
    /** Output of the neuron (after activation) */
    double output;
    /** Derivative of the output of the neuron (after activation with derivative function) */
    double derivative_output;

public:
    /**
     * Default constructor
     * @param input Input of the neuron (already weighted)
     */
    explicit Neuron(double input);
    /**
     * Default destructor
     */
    ~Neuron();

    /**
     * Activate the neuron with the given activation function and store the output and derivative output
     * @param activation_function Activation function
     * @param derivative_activation_function Derivative of the activation function
     */
    void activate(act_func activation_function, act_func derivative_activation_function);

    /**
     * Set the input of the neuron
     * @param new_input Input of the neuron (already weighted)
     */
    void set_input(double new_input);
    /**
     * Get the input of the neuron
     * @return Input of the neuron (already weighted)
     */
    [[nodiscard]] double get_input() const;
    /**
     * Get the output of the neuron
     * @return Output of the neuron (after activation)
     */
    [[nodiscard]] double get_output() const;
    /**
     * Get the derivative output of the neuron
     * @return Output of the neuron (after activation with derivative function)
     */
    [[nodiscard]] double get_derivative_output() const;
};
