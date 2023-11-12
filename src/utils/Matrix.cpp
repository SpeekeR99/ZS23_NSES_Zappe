#include "Matrix.h"

Matrix::Matrix(int rows, int cols, bool randomize) : rows(rows), cols(cols) {
    this->data = std::vector<std::vector<double>>(rows, std::vector<double>(cols, 0));
    if (randomize)
        this->randomize();
}

Matrix::Matrix(int rows, int cols, const std::vector<std::vector<double>> &data) : rows(rows), cols(cols) {
    auto data_deep_copy = std::vector<std::vector<double>>(rows, std::vector<double>(cols, 0));
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            data_deep_copy[i][j] = data[i][j];
    this->data = data_deep_copy;
}

Matrix::Matrix(const Matrix &other) noexcept : rows(other.rows), cols(other.cols) {
    auto data_deep_copy = std::vector<std::vector<double>>(rows, std::vector<double>(cols, 0));
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            data_deep_copy[i][j] = other.data[i][j];
    this->data = data_deep_copy;
}

Matrix::Matrix(Matrix &&other) noexcept : rows(other.rows), cols(other.cols) {
    this->data = std::move(other.data);
    other.rows = 0;
    other.cols = 0;
}

Matrix::~Matrix() = default;

Matrix &Matrix::transpose() {
    std::vector<std::vector<double>> transposed_data(this->cols, std::vector<double>(this->rows, 0));
    for (int i = 0; i < this->rows; i++)
        for (int j = 0; j < this->cols; j++)
            transposed_data[j][i] = this->data[i][j];
    static Matrix transposed_matrix(this->cols, this->rows, transposed_data);
    return transposed_matrix;
}

void Matrix::randomize() {
    std::default_random_engine gen(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> dist(-1, 1);
    for (auto &row : this->data)
        for (auto &col : row)
            col = dist(gen);
}

void Matrix::print() const {
    for (const auto &row : this->data) {
        for (const auto &col : row)
            std::cout << col << " ";
        std::cout << std::endl;
    }
}

void Matrix::set_value(int row, int col, double value) {
    this->data[row][col] = value;
}

void Matrix::set_values(const std::vector<std::vector<double>> &values) {
    this->data = values;
}

double Matrix::get_value(int row, int col) const {
    return this->data[row][col];
}

std::vector<std::vector<double>> Matrix::get_values() const {
    return this->data;
}

Matrix &Matrix::operator=(const Matrix &other) noexcept {
    this->rows = other.rows;
    this->cols = other.cols;
    auto data_deep_copy = std::vector<std::vector<double>>(rows, std::vector<double>(cols, 0));
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            data_deep_copy[i][j] = other.data[i][j];
    this->data = data_deep_copy;
    return *this;
}

Matrix &Matrix::operator=(Matrix &&other) noexcept {
    this->rows = other.rows;
    this->cols = other.cols;
    this->data = std::move(other.data);
    other.rows = 0;
    other.cols = 0;
    return *this;
}

Matrix Matrix::operator*(const Matrix &other) const {
    if (this->cols != other.rows)
        throw std::runtime_error("Matrix multiplication error: incompatible dimensions");
    auto new_data = std::vector<std::vector<double>>(this->rows, std::vector<double>(other.cols, 0));
    for (int i = 0; i < this->rows; i++)
        for (int j = 0; j < other.cols; j++)
            for (int k = 0; k < this->cols; k++)
                new_data[i][j] += this->data[i][k] * other.data[k][j];
    return {this->rows, other.cols, new_data};
}
