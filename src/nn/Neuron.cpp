#include "Neuron.h"

act_func predefined_activation_functions[] = {
        [](double x) -> double { return x; },                   /* linear */
        [](double x) -> double { return x > 0 ? x : 0; },       /* relu */
        [](double x) -> double { return 1 / (1 + exp(-x)); },   /* sigmoid */
        [](double x) -> double { return x > 0 ? 1 : 0; },       /* step */
        [](double x) -> double { return x > 0 ? 1 : -1; },      /* sign */
        [](double x) -> double { return tanh(x); }              /* tanh */
};

Neuron::Neuron(double input) : input(input), output(0), activation_function(nullptr) {
    /* empty */
}

Neuron::Neuron(double input, act_func activation_function) : input(input), output(0),
                                                             activation_function(activation_function) {
    /* empty */
}

Neuron::Neuron(double input, act_func_type activation_function) : input(input), output(0), activation_function(
        predefined_activation_functions[static_cast<int>(activation_function)]) {
    /* empty */
}

Neuron::~Neuron() = default;

void Neuron::activate() {
    this->output = this->activation_function(this->input);
}

void Neuron::set_input(double new_input) {
    this->input = new_input;
}

void Neuron::set_activation_function(act_func new_activation_function) {
    this->activation_function = new_activation_function;
}

void Neuron::set_activation_function(act_func_type new_activation_function) {
    this->activation_function = predefined_activation_functions[static_cast<int>(new_activation_function)];
}

double Neuron::get_output() const {
    return this->output;
}
