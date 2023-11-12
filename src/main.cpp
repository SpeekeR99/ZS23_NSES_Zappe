#include <iostream>
#include "nn/NeuralNetwork.h"
#include "utils/Matrix.h"

int main() {
    std::unique_ptr<NeuralNetwork> nn = std::make_unique<NeuralNetwork>(2, 2, std::vector<uint32_t>{5}, act_func_type::sigmoid);
    nn->set_inputs({1, 2}); /* 1, 2 */
    nn->feed_forward(); /* 4; 0.9820137900379332 */
    auto output = nn->get_output(); /* 5.91006895; 0.9972953351621966 */
    for (auto &o : output)
        std::cout << o << std::endl << std::endl;

    Matrix m1(3, 2, true);
    Matrix m2 = m1.transpose();
    m2.set_value(1, 1, 5);
    std::cout << "m1: " << std::endl;
    m1.print();
    std::cout << std::endl << "m2: " << std::endl;
    m2.print();
    std::cout << std::endl << "m1 * m2: " << std::endl;
    auto m3 = m1 * m2;
    m3.print();
    std::cout << std::endl << "m2 * m1: " << std::endl;
    auto m4 = m2 * m1;
    m4.print();
    std::cout << std::endl;

    return EXIT_SUCCESS;
}
