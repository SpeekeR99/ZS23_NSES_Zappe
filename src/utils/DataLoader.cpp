#include "DataLoader.h"

x_y_pairs DataLoader::load_file(const std::string &filename, uint32_t input_size, uint32_t output_size, char delimiter) {
    x_y_pairs data = {};

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: could not open file " << filename << std::endl;
        return data;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> inputs(input_size);
        std::vector<double> outputs(output_size);

        uint32_t i = 0;
        uint32_t start;
        uint32_t end = 0;
        while ((start = line.find_first_not_of(delimiter, end)) != std::string::npos) {
            end = line.find(delimiter, start);
            std::string token = line.substr(start, end - start);
            std::istringstream iss(token);
            if (i < input_size)
                iss >> inputs[i];
            else
                iss >> outputs[i - input_size];
            i++;
        }

        data.emplace_back(std::make_pair(inputs, outputs));
    }

    return data;
}

x_y_pairs DataLoader::transform_y_to_one_hot(const x_y_pairs &data) {
    std::set<double> unique_outputs;
    for (auto &pair : data)
        unique_outputs.insert(pair.second[0]);
    std::map<double, uint32_t> output_map;
    uint32_t index = 0;
    for (auto &output : unique_outputs)
        output_map[output] = index++;

    uint32_t output_size = unique_outputs.size();
    x_y_pairs one_hot_data = {};

    for (auto &pair : data) {
        std::vector<double> inputs = pair.first;
        std::vector<double> outputs(output_size, 0.);
        outputs[output_map[pair.second[0]]] = 1.;
        one_hot_data.emplace_back(std::make_pair(inputs, outputs));
    }

    return one_hot_data;
}

std::pair<x_y_pairs, x_y_pairs> DataLoader::split_data(const x_y_pairs &data, double train_test_split) {
    auto shuffled_indices = std::vector<uint32_t>(data.size());
    std::iota(shuffled_indices.begin(), shuffled_indices.end(), 0);
    std::shuffle(shuffled_indices.begin(), shuffled_indices.end(), std::mt19937(std::random_device()()));

    auto train_size = static_cast<uint32_t>(data.size() * train_test_split);
    auto test_size = data.size() - train_size;

    x_y_pairs train_data(train_size);
    x_y_pairs test_data(test_size);

    for (uint32_t i = 0; i < train_size; i++)
        train_data[i] = data[shuffled_indices[i]];
    for (uint32_t i = 0; i < test_size; i++)
        test_data[i] = data[shuffled_indices[i + train_size]];

    return std::make_pair(train_data, test_data);
}

x_y_matrix DataLoader::transform_to_matrices(const x_y_pairs &data) {
    Matrix x(data.size(), data[0].first.size(), false);
    Matrix y(data.size(), data[0].second.size(), false);

    for (uint32_t i = 0; i < data.size(); i++) {
        auto pair = data[i];
        x.set_row(i, pair.first);
        y.set_row(i, pair.second);
    }

    return std::make_pair(x, y);
}
