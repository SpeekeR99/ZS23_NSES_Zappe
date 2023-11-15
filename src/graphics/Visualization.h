#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include "../nn/NeuralNetwork.h"
#include "../utils/DataLoader.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_stdlib.h"
#include "imgui_internal.h"

class Visualization {
private:
    GLFWwindow *window;
    const GLFWvidmode *mode;
    bool show_settings_window = false;
    bool show_about_window = false;
    int window_width = 1280;
    int window_height = 720;
    bool fullscreen = false;
    float font_size = 1.0f;

    void init();
    void render();
    void cleanup();

    static void glfw_error_callback(int error, const char *description);

    std::string data_filepath = "data/tren_data1___23.txt";
    int number_of_inputs = 2;
    float data_split_ratio = 0.8f;
    x_y_matrix training_data = std::make_pair(Matrix(0, 0), Matrix(0, 0));
    x_y_matrix test_data = std::make_pair(Matrix(0, 0), Matrix(0, 0));
    int number_of_classes = 0;

    NeuralNetwork nn = NeuralNetwork(0, 0, std::vector<uint32_t>{}, act_func_type::relu, 0.0f, 0, false);
    int number_of_hidden_layers = 1;
    std::vector<int> number_of_neurons_in_hidden_layers = std::vector<int>{8};
    int chosen_activation_function_idx = 1;
    float learning_rate = 0.01f;
    int batch_size = 10;
    bool use_softmax = true;
    int number_of_epochs = 200;

public:
    Visualization();
    ~Visualization();

    void run();
};