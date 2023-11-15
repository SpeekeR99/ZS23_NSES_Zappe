#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_stdlib.h"
#include "imgui_internal.h"

class Visualization {
private:
    GLFWwindow *window;

    void init();
    void render();
    void cleanup();

    static void glfw_error_callback(int error, const char *description);

public:
    Visualization();
    ~Visualization();

    void run();
};