#include "DataLoader.h"

x_y_pairs DataLoader::load_file(const std::string &filename, uint32_t input_size, uint32_t output_size, char delimiter) {
    x_y_pairs data = {};

    std::ifstream file(filename); /* Open file */
    if (!file.is_open()) { /* Check if file is open */
        std::cerr << "Error: could not open file " << filename << std::endl;
        return data;
    }

    std::string line;
    while (std::getline(file, line)) { /* Read file line by line */
        std::vector<double> inputs(input_size);
        std::vector<double> outputs(output_size);

        uint32_t i = 0; /* i is the index of the current input/output */
        uint32_t start; /* start is the index of the start of the current token */
        uint32_t end = 0; /* end is the index of the end of the current token */
        while ((start = line.find_first_not_of(delimiter, end)) != std::string::npos) {
            end = line.find(delimiter, start);
            std::string token = line.substr(start, end - start);
            std::istringstream iss(token);

            if (i < input_size)
                iss >> inputs[i]; /* Read input */
            else
                iss >> outputs[i - input_size]; /* Read output */
            i++;
        }

        data.emplace_back(std::make_pair(inputs, outputs));
    }

    return data;
}

x_y_pairs DataLoader::transform_y_to_one_hot(const x_y_pairs &data) {
    std::set<double> unique_outputs; /* Set of unique outputs */
    for (auto &pair : data)
        unique_outputs.insert(pair.second[0]);

    std::map<double, uint32_t> output_map; /* Map output classes to indices */
    uint32_t index = 0;
    for (auto &output : unique_outputs)
        output_map[output] = index++;

    uint32_t output_size = unique_outputs.size();
    x_y_pairs one_hot_data = {};

    for (auto &pair : data) {
        std::vector<double> inputs = pair.first; /* Inputs are the same */
        std::vector<double> outputs(output_size, 0.); /* Outputs are one-hot encoded */
        outputs[output_map[pair.second[0]]] = 1.;
        one_hot_data.emplace_back(std::make_pair(inputs, outputs));
    }

    return one_hot_data;
}

std::pair<x_y_pairs, x_y_pairs> DataLoader::split_data(const x_y_pairs &data, double train_test_split) {
    /* Shuffle data, so the training and test data are not biased */
    auto shuffled_indices = std::vector<uint32_t>(data.size());
    std::iota(shuffled_indices.begin(), shuffled_indices.end(), 0);
    std::shuffle(shuffled_indices.begin(), shuffled_indices.end(), std::mt19937(std::random_device()()));

    auto train_size = static_cast<uint32_t>(data.size() * train_test_split);
    auto test_size = data.size() - train_size;

    x_y_pairs train_data(train_size);
    x_y_pairs test_data(test_size);

    for (uint32_t i = 0; i < train_size; i++) /* Fill training data */
        train_data[i] = data[shuffled_indices[i]];
    for (uint32_t i = 0; i < test_size; i++) /* Fill test data */
        test_data[i] = data[shuffled_indices[i + train_size]];

    return std::make_pair(train_data, test_data);
}

x_y_matrix DataLoader::transform_to_matrices(const x_y_pairs &data) {
    /* Get input and output sizes */
    Matrix x(data.size(), data[0].first.size(), false);
    Matrix y(data.size(), data[0].second.size(), false);

    for (uint32_t i = 0; i < data.size(); i++) {
        auto pair = data[i];
        x.set_row(i, pair.first); /* Set input row */
        y.set_row(i, pair.second); /* Set output row */
    }

    return std::make_pair(x, y);
}
