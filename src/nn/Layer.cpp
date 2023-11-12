#include "Layer.h"

act_func predefined_activation_functions[] = {
        [](double x) -> double { return x; },                                                       /* linear */
        [](double x) -> double { return x > 0 ? x : 0; },                                           /* relu */
        [](double x) -> double { return 1 / (1 + exp(-x)); },                                       /* sigmoid */
        [](double x) -> double { return x > 0 ? 1 : 0; },                                           /* step */
        [](double x) -> double { return x > 0 ? 1 : -1; },                                          /* sign */
        [](double x) -> double { return tanh(x); },                                                 /* tanh */
};

act_func predefined_derivative_activation_functions[] = {
        [](double x) -> double { return 1; },                                                       /* linear */
        [](double x) -> double { return x > 0 ? 1 : 0; },                                           /* relu */
        [](double x) -> double { return (1 / (1 + exp(-x))) * (1 - (1 / (1 + exp(-x)))); },         /* sigmoid */
        [](double x) -> double { return 0; },                                                       /* step */
        [](double x) -> double { return 0; },                                                       /* sign */
        [](double x) -> double { return 1 - pow(tanh(x), 2); },                                     /* tanh */
};

Layer::Layer(uint32_t size) : size(size), activation_function(nullptr), derivative_activation_function(nullptr) {
    this->neurons.reserve(this->size);
    for (uint32_t i = 0; i < this->size; i++)
        this->neurons.emplace_back(std::make_unique<Neuron>(0));
}

Layer::Layer(uint32_t size, act_func activation_function) : size(size), activation_function(nullptr), derivative_activation_function(nullptr) {
    this->neurons.reserve(this->size);
    for (uint32_t i = 0; i < this->size; i++)
        this->neurons.emplace_back(std::make_unique<Neuron>(0));
    this->set_activation_function(activation_function);
    for (int i = 0; i < static_cast<int>(act_func_type::number_of_activation_functions); i++) {
        if (predefined_activation_functions[i] == activation_function) {
            this->set_derivative_activation_function(predefined_derivative_activation_functions[i]);
            break;
        }
    }
}

Layer::Layer(uint32_t size, act_func_type activation_function) : size(size), activation_function(nullptr), derivative_activation_function(nullptr) {
    this->neurons.reserve(this->size);
    for (uint32_t i = 0; i < this->size; i++)
        this->neurons.emplace_back(std::make_unique<Neuron>(0));
    this->set_activation_function(activation_function);
    this->set_derivative_activation_function(activation_function);
}

Layer::~Layer() = default;

void Layer::activate() {
    for (auto &neuron : this->neurons)
        neuron->activate(this->activation_function, this->derivative_activation_function);
}

void Layer::init_weights(uint32_t rows, uint32_t cols) {
    this->weights = Matrix(rows, cols, true);
}

void Layer::set_inputs(const std::vector<double> &inputs) {
    for (uint32_t i = 0; i < this->size; i++)
        this->neurons[i]->set_input(inputs[i]);
}

void Layer::set_activation_function(act_func new_activation_function) {
    this->activation_function = new_activation_function;
}

void Layer::set_activation_function(act_func_type new_activation_function) {
    this->activation_function = predefined_activation_functions[static_cast<uint32_t>(new_activation_function)];
}

void Layer::set_derivative_activation_function(act_func new_derivative_activation_function) {
    this->derivative_activation_function = new_derivative_activation_function;
}

void Layer::set_derivative_activation_function(act_func_type new_derivative_activation_function) {
    this->derivative_activation_function = predefined_derivative_activation_functions[static_cast<uint32_t>(new_derivative_activation_function)];
}

void Layer::set_weights(Matrix &new_weights) {
    this->weights = new_weights;
}

std::vector<double> Layer::get_output() const {
    std::vector<double> output;
    output.reserve(this->size);
    for (auto &neuron : this->neurons)
        output.push_back(neuron->get_output());
    return output;
}

std::vector<double> Layer::get_softmax_output() const {
    std::vector<double> softmax_output;
    softmax_output.reserve(this->size);
    for (auto &neuron : this->neurons)
        softmax_output.push_back(neuron->get_input());
    double sum = 0;
    for (auto &output : softmax_output)
        sum += exp(output);
    for (auto &output : softmax_output)
        output = exp(output) / sum;
    return softmax_output;
}

std::vector<double> Layer::get_derivative_output() const {
    std::vector<double> derivative_output;
    derivative_output.reserve(this->size);
    for (auto &neuron : this->neurons)
        derivative_output.push_back(neuron->get_derivative_output());
    return derivative_output;
}

uint32_t Layer::get_size() const {
    return this->size;
}

Matrix Layer::get_weights() const {
    return this->weights;
}
