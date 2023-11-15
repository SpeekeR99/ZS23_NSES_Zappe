cmake_minimum_required(VERSION 3.24)
project(ZS23_NSES_Zappe)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenGL REQUIRED)

add_definitions(-DGLEW_STATIC)

# Including GLFW
add_subdirectory(lib/glfw)
include_directories(${GLFW_INCLUDE_DIRS})

# Including GLEW
add_subdirectory(lib/glew)
include_directories(${GLEW_INCLUDE_DIRS})

# Including ImGui
include_directories(lib/imgui)
include_directories(lib/imgui/backends)
include_directories(lib/imgui/misc/cpp)
include_directories(lib/implot)

set(
        imgui_files
        lib/imgui/imconfig.h
        lib/imgui/imgui.cpp
        lib/imgui/imgui.h
        lib/imgui/imgui_draw.cpp
        lib/imgui/imgui_internal.h
        lib/imgui/imgui_tables.cpp
        lib/imgui/imgui_widgets.cpp
        lib/imgui/imstb_rectpack.h
        lib/imgui/imstb_textedit.h
        lib/imgui/imstb_truetype.h
        lib/imgui/misc/cpp/imgui_stdlib.cpp
        lib/imgui/misc/cpp/imgui_stdlib.h
        lib/implot/implot.cpp
        lib/implot/implot.h
        lib/implot/implot_internal.h
        lib/implot/implot_items.cpp
)

set(
        imgui_impl_files
        lib/imgui/backends/imgui_impl_glfw.cpp
        lib/imgui/backends/imgui_impl_glfw.h
        lib/imgui/backends/imgui_impl_opengl3.cpp
        lib/imgui/backends/imgui_impl_opengl3.h
        lib/imgui/backends/imgui_impl_opengl3_loader.h
)


add_executable(
        ZS23_NSES_Zappe
        src/main.cpp
        src/nn/Neuron.cpp
        src/nn/Neuron.h
        src/nn/Layer.cpp
        src/nn/Layer.h
        src/nn/NeuralNetwork.cpp
        src/nn/NeuralNetwork.h
        src/utils/Matrix.cpp
        src/utils/Matrix.h
        src/utils/DataLoader.cpp
        src/utils/DataLoader.h
        src/graphics/Visualization.cpp
        src/graphics/Visualization.h
        ${imgui_files}
        ${imgui_impl_files}
)

target_link_libraries(ZS23_NSES_Zappe glfw libglew_static ${GLFW_LIBRARIES} ${GLEW_LIBRARIES} ${OPENGL_LIBRARY})
