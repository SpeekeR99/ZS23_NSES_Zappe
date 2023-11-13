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

void Layer::set_inputs(const Matrix &inputs) {
    for (uint32_t i = 0; i < this->size; i++)
        this->neurons[i]->set_input(inputs.get_value(i, 0));
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

Matrix Layer::get_output() const {
    Matrix output(this->size, 1, false);
    for (uint32_t i = 0; i < this->size; i++)
        output.set_value(i, 0, this->neurons[i]->get_output());
    return output;
}

Matrix Layer::get_softmax_output() const {
    Matrix softmax_output(this->size, 1, false);
    for (uint32_t i = 0; i < this->size; i++)
        softmax_output.set_value(i, 0, this->neurons[i]->get_input());
    double sum = 0;
    for (uint32_t i = 0; i < this->size; i++)
        sum += exp(softmax_output.get_value(i, 0));
    for (uint32_t i = 0; i < this->size; i++)
        softmax_output.set_value(i, 0, exp(softmax_output.get_value(i, 0)) / sum);
    return softmax_output;
}

Matrix Layer::get_derivative_output() const {
    Matrix derivative_output(this->size, 1, false);
    for (uint32_t i = 0; i < this->size; i++)
        derivative_output.set_value(i, 0, this->neurons[i]->get_derivative_output());
    return derivative_output;
}

Matrix Layer::get_softmax_derivative_output() const {
    Matrix softmax_derivative_output(this->size, this->size, false);
    auto softmax_output = this->get_softmax_output();
    for (uint32_t i = 0; i < this->size; i++)
        for (uint32_t j = 0; j < this->size; j++)
            if (i == j)
                softmax_derivative_output.set_value(i, j, softmax_output.get_value(i, 0) * (1 - softmax_output.get_value(i, 0)));
            else
                softmax_derivative_output.set_value(i, j, -softmax_output.get_value(i, 0) * softmax_output.get_value(j, 0));
    return softmax_derivative_output;
}

uint32_t Layer::get_size() const {
    return this->size;
}

Matrix Layer::get_weights() const {
    return this->weights;
}
