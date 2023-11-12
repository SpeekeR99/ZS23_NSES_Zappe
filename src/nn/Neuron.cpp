#include "Neuron.h"

Neuron::Neuron(double input) : input(input), output(input), derivative_output(input) {
    /* empty */
}

Neuron::~Neuron() = default;

void Neuron::activate(act_func activation_function, act_func derivative_activation_function) {
    this->output = activation_function(this->input);
    this->derivative_output = derivative_activation_function(this->input);
}

void Neuron::set_input(double new_input) {
    this->input = new_input;
}

double Neuron::get_input() const {
    return this->input;
}

double Neuron::get_output() const {
    return this->output;
}

double Neuron::get_derivative_output() const {
    return this->derivative_output;
}
