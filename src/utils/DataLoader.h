#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include "Matrix.h"

/** Vector of pairs of vectors of doubles */
typedef std::vector<std::pair<std::vector<double>, std::vector<double>>> x_y_pairs;
/** Pair of matrices */
typedef std::pair<Matrix, Matrix> x_y_matrix;

/**
 * Class used for loading data from files and transforming it into usable formats
 */
class DataLoader {
public:
    /**
     * Loads data from a file and returns it as a vector of pairs of vectors of doubles
     * @param filename Filepath to the file containing the data
     * @param input_size Number of input features
     * @param output_size Number of output features (expecting 1 basically - class)
     * @param delimiter Delimiter used in the file to separate values
     * @return Vector of pairs of vectors of doubles
     */
    static x_y_pairs load_file(const std::string &filename, uint32_t input_size, uint32_t output_size, char delimiter);
    /**
     * Transforms the basic loaded format to a one-hot encoded format (outputs only)
     * @param data Data in the basic format from load_file function
     * @return Vector of pairs of vectors of doubles (outputs are one-hot encoded)
     */
    static x_y_pairs transform_y_to_one_hot(const x_y_pairs &data);
    /**
     * Splits the data into training and test data
     * @param data Vector of pairs of vectors of doubles (outputs can be one-hot encoded, don't have to be)
     * @param train_test_split Ratio of training data to test data (0.8 means 80 % training data, 20 % test data)
     * @return Pair of vectors of pairs of vectors of doubles (first is training data, second is test data)
     */
    static std::pair<x_y_pairs, x_y_pairs> split_data(const x_y_pairs &data, double train_test_split);
    /**
     * Transforms the data into matrices
     * @param data Vector of pairs of vectors of doubles (outputs can be one-hot encoded, don't have to be)
     * @return Pair of matrices (first matrix is inputs, second matrix is outputs)
     */
    static x_y_matrix transform_to_matrices(const x_y_pairs &data);
};