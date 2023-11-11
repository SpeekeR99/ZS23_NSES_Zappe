#include <iostream>
#include "nn/NeuralNetwork.h"

int main() {
    std::unique_ptr<NeuralNetwork> nn = std::make_unique<NeuralNetwork>(2, 2, std::vector<uint32_t>{5}, act_func_type::sigmoid);
    nn->set_inputs({1, 2}); /* 	0.731058578630074, 0.8807970779779563 */
    nn->forward_propagation();
    /* 0.9316207029716523, 0.9316207029716523, 0.9316207029716523, 0.9316207029716523, 0.9316207029716523 */
    auto output = nn->get_output(); /* 0.9965230039772246,  0.9965230039772246 */
    for (auto &o : output)
        std::cout << o << std::endl;

    return EXIT_SUCCESS;
}
