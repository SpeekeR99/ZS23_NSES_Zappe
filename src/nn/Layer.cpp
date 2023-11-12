#include "Layer.h"

Layer::Layer(uint32_t size) : size(size) {
    this->neurons.reserve(this->size);
    for (uint32_t i = 0; i < this->size; i++)
        this->neurons.emplace_back(std::make_unique<Neuron>(0));
}

Layer::Layer(uint32_t size, act_func activation_function) : size(size) {
    this->neurons.reserve(this->size);
    for (uint32_t i = 0; i < this->size; i++)
        this->neurons.emplace_back(std::make_unique<Neuron>(0, activation_function));
}

Layer::Layer(uint32_t size, act_func_type activation_function) : size(size) {
    this->neurons.reserve(this->size);
    for (uint32_t i = 0; i < this->size; i++)
        this->neurons.emplace_back(std::make_unique<Neuron>(0, activation_function));
}

Layer::~Layer() = default;

void Layer::activate() {
    for (auto &neuron : this->neurons)
        neuron->activate();
}

void Layer::init_weights(uint32_t rows, uint32_t cols) {
    this->weights = Matrix(rows, cols, true);
}

void Layer::set_inputs(const std::vector<double> &inputs) {
    for (uint32_t i = 0; i < this->size; i++)
        this->neurons[i]->set_input(inputs[i]);
}

void Layer::set_activation_function(act_func new_activation_function) {
    for (auto &neuron : this->neurons)
        neuron->set_activation_function(new_activation_function);
}

void Layer::set_activation_function(act_func_type new_activation_function) {
    for (auto &neuron : this->neurons)
        neuron->set_activation_function(new_activation_function);
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
