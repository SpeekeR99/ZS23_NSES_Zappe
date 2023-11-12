#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <chrono>

class Matrix {
private:
    int rows;
    int cols;
    std::vector<std::vector<double>> data;

public:
    Matrix(int rows, int cols, bool randomize = false);
    Matrix(int rows, int cols, const std::vector<std::vector<double>> &data);
    Matrix(const Matrix &other) noexcept;
    Matrix(Matrix &&other) noexcept;
    ~Matrix();

    Matrix &transpose();
    void randomize();
    void print() const;

    void set_value(int row, int col, double value);
    void set_values(const std::vector<std::vector<double>> &values);
    [[nodiscard]] double get_value(int row, int col) const;
    [[nodiscard]] std::vector<std::vector<double>> get_values() const;

    Matrix &operator=(const Matrix &other) noexcept;
    Matrix &operator=(Matrix &&other) noexcept;
    Matrix operator*(const Matrix &other) const;
};
