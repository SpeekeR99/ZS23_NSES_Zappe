#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <chrono>

class Matrix {
private:
    uint32_t rows;
    uint32_t cols;
    std::vector<std::vector<double>> data;

public:
    Matrix(uint32_t rows, uint32_t cols, bool randomize = false);
    Matrix(uint32_t rows, uint32_t cols, const std::vector<std::vector<double>> &data);
    Matrix(const Matrix &other) noexcept;
    Matrix(Matrix &&other) noexcept;
    ~Matrix();

    Matrix &transpose();
    void randomize();
    void print() const;

    void set_value(uint32_t row, uint32_t col, double value);
    void set_row(uint32_t row, const std::vector<double> &values);
    void set_col(uint32_t col, const std::vector<double> &values);
    void set_values(const std::vector<std::vector<double>> &values);
    [[nodiscard]] double get_value(uint32_t row, uint32_t col) const;
    [[nodiscard]] std::vector<double> get_row(uint32_t row) const;
    [[nodiscard]] std::vector<double> get_col(uint32_t col) const;
    [[nodiscard]] std::vector<std::vector<double>> get_values() const;

    Matrix &operator=(const Matrix &other) noexcept;
    Matrix &operator=(Matrix &&other) noexcept;
    Matrix operator*(const Matrix &other) const;
};
