# Neural Network for Classification

This project implements a neural network from scratch in C++ to classify objects described by two features into 5 classes.

## Introduction

This project addresses the task of classifying objects based on two input features using neural networks.
The networks are trained using provided training datasets, and the project includes visualization of training error and classification regions.

## Implementation

The neural networks are implemented in C++.
Key aspects of the implementation include:

*   **Network Structure:** The code allows for the creation of neural networks with multiple layers and configurable number of neurons in each layer.
*   **Training:** The networks are trained on arbitrary datasets using backpropagation.
*   **Activation Functions:** The code includes activation functions such as sigmoid, ReLU, or tanh. The specific choice of activation functions is configurable.
*   **Error Calculation:** The code calculates the error during training. Common error functions include mean squared error or cross-entropy loss.
*   **Backpropagation:** The backpropagation algorithm is used to update the weights of the network during training.

## Usage

### Build

The project is implemented in C++ and can be easily built using `CMake`.

```bat
mkdir build
cd build
cmake ..
cmake --build .
```

```bash
mkdir build
cd build
cmake ..
make
```

### Run

After building the project, you can run the executable to train a neural network on the provided training data.

## File Format

The training data files (`tren_data1.txt` and `tren_data2.txt`) have the following format:

*   Each line represents one training example.
*   Each line contains three space-separated values: `feature1 feature2 class_label`.
    *   `feature1`: The value of the first feature (a floating-point number).
    *   `feature2`: The value of the second feature (a floating-point number).
    *   `class_label`: An integer representing the class to which the object belongs (1 to 5).

Example:

6.9161 -5.7103 1
7.0601 -5.6065 1
7.2404 -5.3848 2
-7.0375 -5.4146 3

## Visualization

The project includes visualization, which is done thanks to the `ImGui` library.
The visualization shows the training error over time and the classification regions learned by the neural network in real time.

The visualization also provides a way to dynamically change the network structure, activation functions, and other parameters at runtime.

## Requirements

*   C++ compiler (e.g., g++)
*   Standard C++ libraries
