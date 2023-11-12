#include <iostream>
#include "nn/NeuralNetwork.h"

int main() {
    std::unique_ptr<NeuralNetwork> nn = std::make_unique<NeuralNetwork>(2, 2, std::vector<uint32_t>{5}, act_func_type::sigmoid);
    nn->set_inputs({1, 2}); /* 1, 2 */
    nn->forward_propagation(); /* 4; 0.9820137900379332 */
    auto output = nn->get_output(); /* 5.91006895; 0.9972953351621966 */
    for (auto &o : output)
        std::cout << o << std::endl;

    return EXIT_SUCCESS;
}
