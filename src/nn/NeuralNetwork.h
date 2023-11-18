#pragma once

#include <iostream>
#include <algorithm>
#include "Layer.h"
#include "../utils/Matrix.h"
#include "../utils/DataLoader.h"

/**
 * Class representing a neural network
 * My neural network serves as an array of layers and it does all the logic behind training and predicting
 * It also stores information for the visualization
 */
class NeuralNetwork {
private:
    /** Size of the input layer */
    uint32_t input_size;
    /** Size of the output layer */
    uint32_t output_size;
    /** Vector of pointers to layers */
    std::vector<std::shared_ptr<Layer>> layers;
    /** Training error */
    Matrix training_error;
    /** Gradient of the neural network */
    std::vector<std::vector<Matrix>> gradient;
    /** Softmax output */
    bool softmax_output;

    /**
     * Set input of the neural network (first layer)
     * @param inputs Inputs to the neural network
     */
    void set_input(const Matrix &inputs);
    /**
     * Get output of the neural network (last layer)
     * @return Output of the neural network
     */
    [[nodiscard]] Matrix get_output() const;

    /**
     * Initialize the weights of the neural network
     * Weights are initialized with random values from -1 to 1
     */
    void init_weights();
    /**
     * Feed forward the neural network
     * Each layer is activated with the given activation function
     */
    void feed_forward();
    /**
     * Reset the gradient of the neural network
     */
    void reset_gradient();
    /**
     * Calculates the loss of the neural network
     * Loss is calculated as MSE or Categorical Cross Entropy depending on the flag softmax_output
     * @param expected_output Expected output of the neural network
     * @return Loss of the neural network (MSE / Categorical Cross Entropy)
     */
    double loss(const Matrix &expected_output);
    /**
     * Back propagate the neural network
     * The true magic happens here :)
     * @param expected_output Expected output of the neural network
     */
    void back_propagation(const Matrix &expected_output);
    /**
     * Update the weights of the neural network based on the gradient and the learning rate
     * @param learning_rate Learning rate
     */
    void update_weights(double learning_rate);

public:
    /**
     * Default constructor
     * @param input_size Number of neurons in the input layer
     * @param output_size Number of neurons in the output layer
     * @param hidden_layers_sizes Number of neurons in each hidden layer
     * @param softmax_output Flag whether to use softmax output or not (MSE / Categorical Cross Entropy)
     */
    NeuralNetwork(uint32_t input_size, uint32_t output_size, const std::vector<uint32_t> &hidden_layers_sizes, bool softmax_output = false);
    /**
     * Constructor with activation function as a function pointer
     * @param input_size Number of neurons in the input layer
     * @param output_size Number of neurons in the output layer
     * @param hidden_layers_sizes Number of neurons in each hidden layer
     * @param activation_function Activation function of the neural network (as a function pointer)
     * @param softmax_output Flag whether to use softmax output or not (MSE / Categorical Cross Entropy)
     */
    NeuralNetwork(uint32_t input_size, uint32_t output_size, const std::vector<uint32_t> &hidden_layers_sizes, act_func activation_function, bool softmax_output = false);
    /**
     * Constructor with activation function as an enum value
     * @param input_size Number of neurons in the input layer
     * @param output_size Number of neurons in the output layer
     * @param hidden_layers_sizes Number of neurons in each hidden layer
     * @param activation_function Activation function of the neural network (as an enum value)
     * @param softmax_output Flag whether to use softmax output or not (MSE / Categorical Cross Entropy)
     */
    NeuralNetwork(uint32_t input_size, uint32_t output_size, const std::vector<uint32_t> &hidden_layers_sizes, act_func_type activation_function, bool softmax_output = false);
    /**
     * Default destructor
     */
    ~NeuralNetwork();

    /**
     * Get the layers of the neural network (as a vector of pointers to layers)
     * @return Layers of the neural network (as a vector of pointers to layers)
     */
    [[nodiscard]] const std::vector<std::shared_ptr<Layer>> &get_layers() const;
    /**
     * Get the training error of the neural network
     * @return Training error of the neural network
     */
    [[nodiscard]] Matrix get_training_error() const;

    /**
     * Train the neural network
     * @param training_data Training data
     * @param epochs Number of epochs
     * @param learning_rate Learning rate
     * @param verbose Flag whether to print the training error after each epoch or not
     * @param min_loss Minimum loss to stop the training process
     * @param delta_loss Minimum delta loss to stop the training process
     */
    void train(x_y_matrix &training_data, uint32_t epochs, double learning_rate, uint32_t batch_size, bool verbose = false, double min_loss = 0.0, double delta_loss = 0.0);
    /**
     * Do one step of the training process
     * @param training_data Training data
     * @param epoch Current epoch
     * @param learning_rate Learning rate
     * @param verbose Flag whether to print the training error after each epoch or not
     */
    void train_one_step(x_y_matrix &training_data, uint32_t epoch, double learning_rate, uint32_t batch_size, bool verbose = false);
    /**
     * Test the neural network
     * @param test_data Test data
     * @return Accuracy of the neural network
     */
    double test(x_y_matrix &test_data);
    /**
     * Predict the output of the neural network for the given inputs
     * @param inputs Inputs to the neural network
     * @return Output of the neural network
     */
    Matrix predict(const Matrix &inputs);

    /**
     * Overload of the assignment operator (copy assignment)
     * @param nn Neural network to copy
     * @return Neural network (this)
     */
    NeuralNetwork &operator=(const NeuralNetwork &nn);
    /**
     * Overload of the bitwise left shift operator (for printing)
     * @param os Output stream
     * @param nn Neural network to print (this)
     * @return Output stream
     */
    friend std::ostream &operator<<(std::ostream &os, const NeuralNetwork &nn);
};
