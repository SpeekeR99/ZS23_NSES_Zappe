#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "nn/NeuralNetwork.h"
#include "utils/Matrix.h"
#include "utils/DataLoader.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_stdlib.h"
#include "imgui_internal.h"

const std::string data_filepath = "data/tren_data1___23.txt";
const uint32_t number_of_input_features = 2;
const uint32_t number_of_output_features = 1;
const uint32_t number_of_classes = 5;

/**
 * Callback function for GLFW errors
 * @param error Error code
 * @param description Description of the error
 */
static void glfw_error_callback(int error, const char *description) {
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

int main() {
    std::unique_ptr<NeuralNetwork> nn = std::make_unique<NeuralNetwork>(
            number_of_input_features,
            number_of_classes,
            std::vector<uint32_t>{8},
            act_func_type::relu,
            0.01,
            10,
            true
    );

    x_y_pairs training_data_temp = DataLoader::load_file(
            data_filepath,
            number_of_input_features,
            number_of_output_features,
            ' '
    );
    training_data_temp = DataLoader::transform_y_to_one_hot(training_data_temp);
    x_y_matrix training_data = DataLoader::transform_to_matrices(training_data_temp);

    nn->train(training_data, 200, true);

    for (auto &pair : training_data_temp) {
        Matrix input = Matrix(1, number_of_input_features, {pair.first});
        Matrix output = Matrix(1, number_of_classes, {pair.second});
        Matrix prediction = nn->predict(input);
        std::cout << "Input: " << input << std::endl;
        std::cout << "Expected output: " << output << std::endl;
        std::cout << "Prediction: " << prediction.transpose() << std::endl;
        std::cout << std::endl;
    }

    // Initialize the library
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return EXIT_FAILURE;

    // GL 3.0 + GLSL 330
    const char *glsl_version = "#version 330";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    // Create a windowed mode window and its OpenGL context
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); // Disable resizing
    GLFWwindow *window = glfwCreateWindow(800, 600, "Maze", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return EXIT_FAILURE;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);
    const GLFWvidmode *mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
    glfwSetWindowPos(window, (mode->width - 800) / 2, (mode->height - 600) / 2);

    // Initialize GLEW
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void) io;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Print out some info about the graphics drivers
    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLEW version: " << glewGetString(GLEW_VERSION) << std::endl;

    // Set callbacks for user inputs
//    glfwSetCursorPosCallback(window, cursor_position_callback);

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Background color
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window)) {
        // Poll for and process events
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Render here
        glClear(GL_COLOR_BUFFER_BIT);

        // GUI part
        {
            // Set up style variables
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);

            // Set up the main GUI configuration window
            ImGui::Begin("Configuration", nullptr,
                         ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                         ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |
                         ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoTitleBar);

            // Set up the main GUI window position, size and font size
            ImGui::SetWindowPos(ImVec2(0, 0));

            // Set up the main GUI menu bar
            if (ImGui::BeginMenuBar()) {
                if (ImGui::BeginMenu("File")) { // File menu
                    if (ImGui::MenuItem("Export as Image (with GUI)"))
                        ;
                    if (ImGui::MenuItem("Export as Image (without GUI)"))
                        ;
                    if (ImGui::MenuItem("Export as Image (raw maze)"))
                        ;
                    ImGui::Separator();
                    if (ImGui::MenuItem("Exit", "Alt+F4"))
                        glfwSetWindowShouldClose(window, true);
                    ImGui::EndMenu();
                }
                if (ImGui::BeginMenu("Settings")) { // Settings menu
                    if (ImGui::MenuItem("Graphics Settings")) {
                    }
                    ImGui::EndMenu();
                }
                if (ImGui::BeginMenu("Help")) { // Help menu
                    if (ImGui::MenuItem("About")) {
                    }
                    ImGui::EndMenu();
                }
                ImGui::EndMenuBar();
            }

            ImGui::End();

            // Reset the style
            ImGui::PopStyleVar(2);
        }

        // ImGui Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Swap front and back buffers
        glfwSwapBuffers(window);
        glfwSwapInterval(1); // Enable vsync
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return EXIT_SUCCESS;
}
