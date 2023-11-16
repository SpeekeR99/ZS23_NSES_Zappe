#pragma once

#include <vector>
#include <memory>
#include <random>
#include <chrono>
#include "Neuron.h"
#include "../utils/Matrix.h"

/** Activation function type */
enum class act_func_type {
    linear = 0,
    relu,
    sigmoid,
    step,
    sign,
    tanh,
    number_of_activation_functions /* Enum trick to get the number of activation functions */
};

/**
 * Class representing a layer
 * My layer serves as an array of neurons and it does all the logic behind activations, derivations and weights
 */
class Layer {
private:
    /** Size of the layer (number of neurons) */
    uint32_t size;
    /** Vector of neurons */
    std::vector<std::unique_ptr<Neuron>> neurons;
    /** Activation function of the layer */
    act_func activation_function;
    /** Derivative of the activation function of the layer */
    act_func derivative_activation_function;
    /** Weights of the layer (weights include bias term) */
    Matrix weights = Matrix(0, 0);

public:
    /**
     * Default constructor
     * @param size Size of the layer (number of neurons)
     */
    explicit Layer(uint32_t size);
    /**
     * Constructor with activation function
     * @param size Size of the layer (number of neurons)
     * @param activation_function Activation function of the layer (as a function pointer)
     */
    Layer(uint32_t size, act_func activation_function);
    /**
     * Constructor with activation function
     * @param size Size of the layer (number of neurons)
     * @param activation_function Activation function of the layer (as an enum value)
     */
    Layer(uint32_t size, act_func_type activation_function);
    /**
     * Default destructor
     */
    ~Layer();

    /**
     * Activate the layer (each neuron) with the given activation function
     */
    void activate();

    /**
     * Initialize the weights of the layer with random values from -1 to 1
     * @param rows Rows of the weights matrix (number of neurons in this layer)
     * @param cols Columns of the weights matrix (number of neurons in the previous layer + 1 for the bias term)
     */
    void init_weights(uint32_t rows, uint32_t cols);
    /**
     * Set the inputs of the layer (each neuron)
     * @param inputs Inputs of the layer (each neuron)
     */
    void set_inputs(const Matrix &inputs);
    /**
     * Set the activation function of the layer
     * @param new_activation_function Activation function of the layer (as a function pointer)
     */
    void set_activation_function(act_func new_activation_function);
    /**
     * Set the activation function of the layer
     * @param new_activation_function Activation function of the layer (as an enum value)
     */
    void set_activation_function(act_func_type new_activation_function);
    /**
     * Set the derivative of the activation function of the layer
     * @param new_derivative_activation_function Derivative of the activation function of the layer (as a function pointer)
     */
    void set_derivative_activation_function(act_func new_derivative_activation_function);
    /**
     * Set the derivative of the activation function of the layer
     * @param new_derivative_activation_function Derivative of the activation function of the layer (as an enum value)
     */
    void set_derivative_activation_function(act_func_type new_derivative_activation_function);
    /**
     * Set the weights of the layer
     * @param new_weights Weights of the layer (weights include bias term)
     */
    void set_weights(Matrix &new_weights);
    /**
     * Get the output of the layer (each neuron)
     * @return Output of the layer (each neuron)
     */
    [[nodiscard]] Matrix get_output() const;
    /**
     * Get the softmax output of the layer (each neuron)
     * @return Softmax output of the layer (each neuron)
     */
    [[nodiscard]] Matrix get_softmax_output() const;
    /**
     * Get the derivative output of the layer (each neuron)
     * @return Derivative output of the layer (each neuron)
     */
    [[nodiscard]] Matrix get_derivative_output() const;
    /**
     * Get the softmax derivative output of the layer (each neuron)
     * @return Softmax derivative output of the layer (each neuron)
     */
    [[nodiscard]] Matrix get_softmax_derivative_output() const;
    /**
     * Get the size of the layer (number of neurons)
     * @return Size of the layer (number of neurons)
     */
    [[nodiscard]] uint32_t get_size() const;
    /**
     * Get the weights of the layer (weights include bias term)
     * @return Weights of the layer (weights include bias term)
     */
    [[nodiscard]] Matrix get_weights() const;
};
