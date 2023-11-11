#include <iostream>
#include "nn/NeuralNetwork.h"

int main() {
    std::unique_ptr<NeuralNetwork> nn = std::make_unique<NeuralNetwork>(2, 1, std::vector<uint32_t>{10, 10}, act_func_type::sigmoid);
    std::unique_ptr<Neuron> neuron = std::make_unique<Neuron>(0.5, act_func_type::sigmoid);
    neuron->activate();
    std::cout << neuron->get_output() << std::endl;
    neuron->set_activation_function(act_func_type::relu);
    neuron->activate();
    std::cout << neuron->get_output() << std::endl;
    return 0;
}
