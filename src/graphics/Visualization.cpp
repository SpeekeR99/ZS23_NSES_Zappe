#include "Visualization.h"

void Visualization::glfw_error_callback(int error, const char *description) {
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

Visualization::Visualization() : window(nullptr), mode(nullptr) {
    this->init();
}

Visualization::~Visualization() {
    this->cleanup();
}

void Visualization::init() {
    /* Initialize the library */
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        throw std::runtime_error("GLFW initialization error");

    /* GL 3.0 + GLSL 330 */
    const char *glsl_version = "#version 330";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    /* Create a windowed mode window and its OpenGL context */
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); /* Disable resizing */
    this->window = glfwCreateWindow(window_width, window_height, "NSES Zappe", nullptr, nullptr);
    if (!this->window) {
        glfwTerminate();
        throw std::runtime_error("GLFW window creation error");
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(this->window);
    mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
    glfwSetWindowPos(this->window, (mode->width - window_width) / 2, (mode->height - window_height) / 2);

    /* Initialize GLEW */
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
        throw std::runtime_error("GLEW initialization error");
    }

    /* Setup Dear ImGui context */
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void) io;

    /* Setup Dear ImGui style */
    ImGui::StyleColorsDark();

    /* Print out some info about the graphics drivers */
    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLEW version: " << glewGetString(GLEW_VERSION) << std::endl;

    /* Setup Platform/Renderer backends */
    ImGui_ImplGlfw_InitForOpenGL(this->window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    /* Background color */
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
}

void Visualization::render() {
    /* Poll for and process events */
    glfwPollEvents();

    /* Start the Dear ImGui frame */
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    /* Render here */
    glClear(GL_COLOR_BUFFER_BIT);

    /* Training part */
    if (training) {
        /* Do one step of training */
        nn.train_one_step(training_data, current_epoch++, learning_rate, batch_size, true);

        /* Clear cached data */
        visuals_data_x_nn_classified.clear();
        visuals_data_y_nn_classified.clear();
        visuals_data_class_nn_classified.clear();

        /* Classify the space */
        int steps = 50;
        for (int i = 0; i < steps; i++) {
            for (int j = 0; j < steps; ++j) {
                float x = x_min + (x_max - x_min) / (float) steps * i;
                float y = y_min + (y_max - y_min) / (float) steps * j;

                Matrix input = Matrix(1, 2);
                input.set_value(0, 0, x);
                input.set_value(0, 1, y);

                Matrix output = nn.predict(input);

                visuals_data_x_nn_classified.emplace_back(x);
                visuals_data_y_nn_classified.emplace_back(y);
                visuals_data_class_nn_classified.emplace_back(static_cast<int>(output.argmax()));
            }
        }

        /* Check if the training is finished */
        if (current_epoch > number_of_epochs) {
            training = false;
            std::cout << "Training finished" << std::endl;
            std::cout << "Test data accuracy: " << nn.test(test_data) * 100 << " %" << std::endl;
        }
    }

    /* GUI part */
    {
        /* Set up style variables */
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);

        /* Set up the main GUI configuration window */
        ImGui::Begin("Configuration", nullptr,
                     ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |
                     ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoTitleBar);

        /* Set up the main GUI window position, size and font size */
        ImGui::SetWindowPos(ImVec2(0, 0));
        ImGui::SetWindowSize(ImVec2((float) window_width / 2.0f, (float) window_height));
        ImGui::SetWindowFontScale(font_size);

        /* Set up the main GUI menu bar */
        if (ImGui::BeginMenuBar()) {
            if (ImGui::BeginMenu("File")) { /* File menu */
                if (ImGui::MenuItem("Exit", "Alt+F4"))
                    glfwSetWindowShouldClose(window, true);
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Settings")) { /* Settings menu */
                if (ImGui::MenuItem("Graphics Settings")) {
                    show_settings_window = true;
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Help")) { /* Help menu */
                if (ImGui::MenuItem("About")) {
                    show_about_window = true;
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }

        /* Data section */
        ImGui::SeparatorText("Data");

        if (ImGui::InputText("Path to file with data", &data_filepath)) {

        }

        if (ImGui::InputInt("Number of inputs per sample", &number_of_inputs, 1, 1)) {

        }

        if (ImGui::InputFloat("Data split ratio", &data_split_ratio, 0.1f, 0.3f, "%.1f")) {
            /* On change, clamp the value between 0.0 and 1.0 */
            if (data_split_ratio < 0.0f) data_split_ratio = 0.0f;
            if (data_split_ratio > 1.0f) data_split_ratio = 1.0f;
        }

        if (ImGui::Button("Load data")) {
            /* Load the data */
            x_y_pairs data_temp = DataLoader::load_file(data_filepath, number_of_inputs, 1, ' ');
            /* Transform the data to one-hot encoding */
            data_temp = DataLoader::transform_y_to_one_hot(data_temp);
            /* Split the data into training and test data */
            auto data = DataLoader::split_data(data_temp, data_split_ratio);
            this->training_data = DataLoader::transform_to_matrices(data.first);
            this->test_data = DataLoader::transform_to_matrices(data.second);

            std::cout << "Training data size: " << training_data.first.get_dims()[0] << std::endl;
            std::cout << "Test data size: " << test_data.first.get_dims()[0] << std::endl;
            number_of_classes = static_cast<int>(training_data.second.get_dims()[1]);
            std::cout << "Number of classes: " << number_of_classes << std::endl;

            /* Clear cached data */
            visuals_data_x.clear();
            visuals_data_y.clear();
            visuals_data_class.clear();
            visuals_data_x_nn_classified.clear();
            visuals_data_y_nn_classified.clear();
            visuals_data_class_nn_classified.clear();

            /* Prepare the data for visualization */
            for (int i = 0; i < training_data.first.get_dims()[0]; i++) {
                visuals_data_x.emplace_back(training_data.first.get_row(i).get_values()[0][0]);
                visuals_data_y.emplace_back(training_data.first.get_row(i).get_values()[0][1]);
                visuals_data_class.emplace_back(static_cast<int>(training_data.second.get_row(i).argmax()));
            }
            for (int i = 0; i < test_data.first.get_dims()[0]; i++) {
                visuals_data_x.emplace_back(test_data.first.get_row(i).get_values()[0][0]);
                visuals_data_y.emplace_back(test_data.first.get_row(i).get_values()[0][1]);
                visuals_data_class.emplace_back(static_cast<int>(test_data.second.get_row(i).argmax()));
            }

            x_min = std::min_element(visuals_data_x.begin(), visuals_data_x.end()).operator*();
            x_max = std::max_element(visuals_data_x.begin(), visuals_data_x.end()).operator*();
            y_min = std::min_element(visuals_data_y.begin(), visuals_data_y.end()).operator*();
            y_max = std::max_element(visuals_data_y.begin(), visuals_data_y.end()).operator*();
            auto x_range = x_max - x_min;
            auto y_range = y_max - y_min;
            x_min -= x_range * 0.1f;
            x_max += x_range * 0.1f;
            y_min -= y_range * 0.1f;
            y_max += y_range * 0.1f;

            std::cout << "Data loaded" << std::endl;
        }

        /* Neural Network settings section */
        ImGui::SeparatorText("Neural Network settings");

        int temp_old_number_of_hidden_layers = number_of_hidden_layers;
        if (ImGui::InputInt("Number of hidden layers", &number_of_hidden_layers, 1, 1)) {
            if (number_of_hidden_layers < 0) number_of_hidden_layers = 0;
            number_of_neurons_in_hidden_layers.resize(number_of_hidden_layers);
            if (temp_old_number_of_hidden_layers < number_of_hidden_layers)
                chosen_activation_functions.emplace(chosen_activation_functions.begin() + temp_old_number_of_hidden_layers, 1);
            else {
                auto last_hidden_layer = chosen_activation_functions.end() - 2;
                chosen_activation_functions.erase(last_hidden_layer);
            }
        }

        const char *activation_function_list[] = {
                "Linear",
                "ReLU",
                "Sigmoid",
                "Step",
                "Sign",
                "Tanh"
        };
        for (int i = 0; i < number_of_hidden_layers; i++) {
            std::string label = "Number of neurons in layer " + std::to_string(i + 1);
            if (ImGui::InputInt(label.c_str(), &number_of_neurons_in_hidden_layers[i], 1, 1)) {
                if (number_of_neurons_in_hidden_layers[i] < 0) number_of_neurons_in_hidden_layers[i] = 0;
            }

            std::string label2 = "Layer " + std::to_string(i + 1) + " activation function";
            if (ImGui::Combo(label2.c_str(), (int *) &chosen_activation_functions[i], activation_function_list, IM_ARRAYSIZE(activation_function_list))) {

            }
        }
        if (ImGui::Combo("Output activation function", (int *) &chosen_activation_functions[number_of_hidden_layers], activation_function_list, IM_ARRAYSIZE(activation_function_list))) {

        }

        if (ImGui::Checkbox("Use softmax in output layer instead", &use_softmax)) {

        }

        if (ImGui::Button("Create Neural Network")) {
            /* Create the neural network */
            std::vector<uint32_t> temp_vector{};
            temp_vector.resize(number_of_hidden_layers);
            for (int i = 0; i < number_of_hidden_layers; i++)
                temp_vector[i] = number_of_neurons_in_hidden_layers[i];

            nn = NeuralNetwork(number_of_inputs, number_of_classes, temp_vector, use_softmax);
            for (int i = 0; i < number_of_hidden_layers + 1; i++)
                nn.get_layers()[i + 1]->set_activation_function(static_cast<act_func_type>(chosen_activation_functions[i]));

            std::cout << "Neural Network created" << std::endl;
            std::cout << nn << std::endl;

            /* Clear cached data */
            visuals_data_x_nn_classified.clear();
            visuals_data_y_nn_classified.clear();
            visuals_data_class_nn_classified.clear();
        }

        /* Training section */
        ImGui::SeparatorText("Training");

        if (ImGui::InputInt("Number of epochs", &number_of_epochs, 1, 5)) {

        }

        if (ImGui::InputFloat("Learning rate", &learning_rate, 0.001f, 0.01f, "%.5f")) {

        }

        if (ImGui::InputInt("Batch size", &batch_size, 1, 5)) {
            if (batch_size < 1) batch_size = 1;
        }

        if (ImGui::Button("Train")) {
            std::cout << "Training started" << std::endl;
            training = true;
            current_epoch = 1;
        }

        if (ImGui::Button("Pause")) {
            std::cout << "Training paused" << std::endl;
            training = false;
        }
        ImGui::SameLine();
        if (ImGui::Button("Resume")) {
            std::cout << "Training resumed" << std::endl;
            training = true;
        }

        ImGui::Separator();

        if (ImGui::Button("Reset")) {
            /* Reset the neural network and training */
            std::cout << "Training stopped; Neural Network reset" << std::endl;
            training = false;
            current_epoch = 1;

            std::vector<uint32_t> temp_vector{};
            temp_vector.resize(number_of_hidden_layers);
            for (int i = 0; i < number_of_hidden_layers; i++)
                temp_vector[i] = number_of_neurons_in_hidden_layers[i];

            nn = NeuralNetwork(number_of_inputs, number_of_classes, temp_vector, use_softmax);
            for (int i = 0; i < number_of_hidden_layers + 1; i++)
                nn.get_layers()[i + 1]->set_activation_function(static_cast<act_func_type>(chosen_activation_functions[i]));

            /* Clear cached data */
            visuals_data_x_nn_classified.clear();
            visuals_data_y_nn_classified.clear();
            visuals_data_class_nn_classified.clear();
        }

        ImGui::End();

        /* Right side of the GUI, data visualization, loss function */
        ImGui::Begin("Data visuals", nullptr,
                     ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |
                     ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoTitleBar);

        /* Set up the main GUI window position, size and font size */
        ImGui::SetWindowPos(ImVec2((float) window_width / 2.0f, 0));
        ImGui::SetWindowSize(ImVec2((float) window_width / 2.0f, (float) window_height));
        ImGui::SetWindowFontScale(font_size);

        /* TabBar */
        if (ImGui::BeginTabBar("##TabBar")) {
            /* Data tab */
            if (ImGui::BeginTabItem("Data")) {
                /* Visualization of the data and the neural network classification of the space */
                ImPlot::SetNextAxesLimits(-10.0f, 10.0f, -10.0f, 10.0f, ImGuiCond_Appearing);
                if (ImPlot::BeginPlot("Data", "x", "y", ImVec2(-1, -1), 0)) {
                    /* Plot the data */
                    for (int i = 0; i < visuals_data_class.size(); i++) {
                        /* Get the right color */
                        int class_idx;
                        for (int j = 0; j < number_of_classes; j++) {
                            if (visuals_data_class[i] == j) {
                                class_idx = j;
                                break;
                            }
                        }

                        /* Test data should be darker */
                        auto color = ImPlot::GetColormapColor(class_idx);
                        if (i >= training_data.first.get_dims()[0]) {
                            color.x *= 0.5f;
                            color.y *= 0.5f;
                            color.z *= 0.5f;
                        }

                        /* Setup the style */
                        ImPlot::PushStyleColor(ImPlotCol_MarkerOutline, color);
                        ImPlot::PushStyleColor(ImPlotCol_MarkerFill, color);
                        ImPlot::PushStyleVar(ImPlotStyleVar_Marker, ImPlotMarker_Circle);
                        ImPlot::PushStyleVar(ImPlotStyleVar_MarkerSize, 4.0f);
                        ImPlot::PushStyleVar(ImPlotStyleVar_MarkerWeight, 2.0f);

                        /* Plot the data */
                        ImPlot::PlotScatter(("Class " + std::to_string(class_idx)).c_str(), &visuals_data_x[i],
                                            &visuals_data_y[i], 1);

                        /* Reset the style */
                        ImPlot::PopStyleVar(3);
                        ImPlot::PopStyleColor(2);
                    }

                    /* Now plot with x % alpha the neural network classification of the space */
                    for (int i = 0; i < visuals_data_class_nn_classified.size(); i++) {
                        /* Get the right color */
                        int class_idx;
                        for (int j = 0; j < number_of_classes; j++) {
                            if (visuals_data_class_nn_classified[i] == j) {
                                class_idx = j;
                                break;
                            }
                        }

                        /* Setup the style */
                        auto color = ImPlot::GetColormapColor(class_idx);
                        color.w = 0.1f;
                        ImPlot::PushStyleColor(ImPlotCol_MarkerOutline, color);
                        ImPlot::PushStyleColor(ImPlotCol_MarkerFill, color);
                        ImPlot::PushStyleVar(ImPlotStyleVar_Marker, ImPlotMarker_Circle);
                        ImPlot::PushStyleVar(ImPlotStyleVar_MarkerSize, 10.0f);
                        ImPlot::PushStyleVar(ImPlotStyleVar_MarkerWeight, 1.0f);

                        /* Plot the data */
                        ImPlot::PlotScatter(("##Classification " + std::to_string(i)).c_str(), &visuals_data_x_nn_classified[i],
                                            &visuals_data_y_nn_classified[i], 1);

                        /* Reset the style */
                        ImPlot::PopStyleVar(3);
                        ImPlot::PopStyleColor(2);
                    }
                }

                ImPlot::EndPlot();
                ImGui::EndTabItem();
            }

            /* Loss function tab */
            if (ImGui::BeginTabItem("Loss function")) {
                /* Loss function over time */
                Matrix training_error = nn.get_training_error();
                float data[training_error.get_dims()[0]];
                for (int i = 0; i < training_error.get_dims()[0]; i++)
                    data[i] = training_error.get_row(i).get_values()[0][0];

                /* Plot the function */
                ImPlot::SetNextAxesLimits(0.0f, (float) training_error.get_dims()[0], 0.0f, 1.0f, ImGuiCond_Appearing);
                if (ImPlot::BeginPlot("Loss function", "Epoch", "Loss", ImVec2(-1, -1), 0, ImPlotAxisFlags_AutoFit)) {
                    ImPlot::PlotLine("Loss", data, training_error.get_dims()[0]);
                    ImPlot::EndPlot();
                }

                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }

        ImGui::End();

        /* Reset the style */
        ImGui::PopStyleVar(2);
    }

    /* Settings window */
    {
        if (show_settings_window) {
            /* Create a window */
            ImGui::Begin("Graphics Settings", &show_settings_window,
                         ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);

            /* Set the window size */
            ImGui::SetWindowSize(ImVec2(600, 400));

            /* Window settings section */
            ImGui::SeparatorText("Window Settings");

            /* Set the size of the window and the position of the window */
            if (ImGui::Button("Set 1280x720")) {
                window_width = 1280;
                window_height = 720;
                glfwSetWindowSize(window, window_width, window_height);
                glfwSetWindowPos(window, (mode->width - window_width) / 2, (mode->height - window_height) / 2);
            }
            ImGui::SameLine();
            if (ImGui::Button("Set 1600x900")) {
                window_width = 1600;
                window_height = 900;
                glfwSetWindowSize(window, window_width, window_height);
                glfwSetWindowPos(window, (mode->width - window_width) / 2, (mode->height - window_height) / 2);
            }
            ImGui::SameLine();
            if (!fullscreen) {
                if (ImGui::Button("Fullscreen")) {
                    window_width = mode->width;
                    window_height = mode->height;
                    fullscreen = true; // Set fullscreen to true
                }
            } else {
                if (ImGui::Button("Windowed")) {
                    window_width = 1280;
                    window_height = 720;
                    glfwSetWindowMonitor(window, nullptr, 0, 0, window_width, window_height, 0);
                    glfwSetWindowPos(window, (mode->width - window_width) / 2, (mode->height - window_height) / 2);
                    fullscreen = false; // Set fullscreen to false
                }
            }

            /* Set the font size */
            if (ImGui::InputFloat("Font Size", &font_size, 0.1f, 0.3f, "%.1f")) {
                // On change, clamp the value between 1.0 and 2.0
                if (font_size < 1.0f) font_size = 1.0f;
                if (font_size > 2.0f) font_size = 2.0f;
            }
            ImGui::SetWindowFontScale(font_size);

            /* Colors section */
            ImGui::SeparatorText("Colors");
            static int style_idx = 0; // Overall style
            if (ImGui::Combo("Style", &style_idx, "Dark\0Light\0Classic\0")) {
                switch (style_idx) {
                    case 0:
                        ImGui::StyleColorsDark();
                        break;
                    case 1:
                        ImGui::StyleColorsLight();
                        break;
                    case 2:
                        ImGui::StyleColorsClassic();
                        break;
                    default:
                        break;
                }
            }
            ImGui::End();
        }
    }

    /* About Window */
    {
        if (show_about_window) {
            ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Always,
                                    ImVec2(0.5f, 0.5f)); // Center the window
            ImGui::Begin("About", &show_about_window,
                         ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
                         ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoCollapse);
            ImGui::SetWindowFontScale(font_size); // Set the font size
            ImGui::Text("This application was made by:\nDominik Zappe");
            ImGui::Separator();
            ImGui::Text("Application serves as Semestral Project for NSES");
            ImGui::End();
        }
    }

    /* ImGui Rendering */
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(this->window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    /* Swap front and back buffers */
    glfwSwapBuffers(this->window);
    glfwSwapInterval(1); /* Enable vsync */
}

void Visualization::cleanup() {
    /* Cleanup */
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

void Visualization::run() {
    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window)) {
        this->render();
    }
}
