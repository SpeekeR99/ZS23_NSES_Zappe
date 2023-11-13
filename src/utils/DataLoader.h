#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include "Matrix.h"

typedef std::vector<std::pair<std::vector<double>, std::vector<double>>> x_y_pairs;
typedef std::pair<Matrix, Matrix> x_y_matrix;

class DataLoader {
public:
    static x_y_pairs load_file(const std::string &filename, uint32_t input_size, uint32_t output_size, char delimiter);
    static x_y_pairs transform_y_to_one_hot(const x_y_pairs &data);
    static x_y_matrix transform_to_matrices(const x_y_pairs &data);
};