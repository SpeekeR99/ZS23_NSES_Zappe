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
#include "implot.h"
#include "implot_internal.h"

/**
 * Class representing the visualization
 * This class serves as the true main function basically
 * Uses OpenGL, GLFW, GLEW and ImGui (with ImPlot)
 */
class Visualization {
private:
    /** Window to render to */
    GLFWwindow *window;
    /** Video mode of the window */
    const GLFWvidmode *mode;
    /** Settings popup window */
    bool show_settings_window = false;
    /** About popup window */
    bool show_about_window = false;
    /** Window width */
    int window_width = 1280;
    /** Window height */
    int window_height = 720;
    /** Fullscreen flag */
    bool fullscreen = false;
    /** Font size */
    float font_size = 1.0f;

    /**
     * Initialize the visualization (OpenGL, GLFW, GLEW, ImGui, ImPlot, callbacks, videomode...)
     */
    void init();
    /**
     * Render the visualization (ImGui, ImPlot)
     * This is the main loop
     */
    void render();
    /**
     * Cleanup the visualization (OpenGL, GLFW, GLEW, ImGui, ImPlot)
     */
    void cleanup();

    /**
     * GLFW error callback
     * @param error Error code
     * @param description Error description
     */
    static void glfw_error_callback(int error, const char *description);

    /** Data filepath, can be changed from the gui */
    std::string data_filepath = "data/tren_data1___23.txt";
    /** Number of inputs, can be changed from the gui */
    int number_of_inputs = 2;
    /** Data split ratio, can be changed from the gui */
    float data_split_ratio = 0.8f;
    /** Training data, obtained from DataLoader */
    x_y_matrix training_data = std::make_pair(Matrix(0, 0), Matrix(0, 0));
    /** Test data, obtained from DataLoader */
    x_y_matrix test_data = std::make_pair(Matrix(0, 0), Matrix(0, 0));
    /** Number of classes, obtained from DataLoader */
    int number_of_classes = 0;

    /** Cache for the data visualization */
    std::vector<float> visuals_data_x{};
    /** Cache for the data visualization */
    float x_min = 0.0f;
    /** Cache for the data visualization */
    float x_max = 0.0f;
    /** Cache for the data visualization */
    std::vector<float> visuals_data_y{};
    /** Cache for the data visualization */
    float y_min = 0.0f;
    /** Cache for the data visualization */
    float y_max = 0.0f;
    /** Cache for the data visualization */
    std::vector<int> visuals_data_class{};

    /** Neural network, can be changed from the gui */
    NeuralNetwork nn = NeuralNetwork(0, 0, std::vector<uint32_t>{}, act_func_type::relu, false);
    /** Number of hidden layers, can be changed from the gui */
    int number_of_hidden_layers = 1;
    /** Number of neurons in hidden layers, can be changed from the gui */
    std::vector<int> number_of_neurons_in_hidden_layers = std::vector<int>{8};
    /** Activation function, can be changed from the gui */
    std::vector<int> chosen_activation_functions = std::vector<int>{static_cast<int>(act_func_type::relu), static_cast<int>(act_func_type::relu)};
    /** Learning rate, can be changed from the gui */
    float learning_rate = 0.01f;
    /** Batch size, can be changed from the gui */
    int batch_size = 10;
    /** Use softmax flag, can be changed from the gui */
    bool use_softmax = true;

    /** Cache for the nn visualization */
    std::vector<float> visuals_data_x_nn_classified{};
    /** Cache for the nn visualization */
    std::vector<float> visuals_data_y_nn_classified{};
    /** Cache for the nn visualization */
    std::vector<int> visuals_data_class_nn_classified{};

    /** Number of epochs, can be changed from the gui */
    int number_of_epochs = 200;
    /** Current epoch of the training process */
    int current_epoch = 1;
    /** Minimum loss to stop the training process, can be changed from the gui */
    double min_loss = 0.0;
    /** Minimum delta loss to stop the training process, can be changed from the gui */
    double delta_loss = 0.0;
    /** Flag if the training is running */
    bool training = false;

public:
    /**
     * Default constructor, calls init()
     */
    Visualization();
    /**
     * Default destructor, calls cleanup()
     */
    ~Visualization();

    /**
     * Main loop, calls render()
     */
    void run();
};