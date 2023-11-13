#include "Matrix.h"

Matrix::Matrix(uint32_t rows, uint32_t cols, bool randomize) : rows(rows), cols(cols) {
    this->data = std::vector<std::vector<double>>(rows, std::vector<double>(cols, 0));
    if (randomize)
        this->randomize();
}

Matrix::Matrix(uint32_t rows, uint32_t cols, const std::vector<std::vector<double>> &data) : rows(rows), cols(cols) {
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

Matrix Matrix::transpose() const {
    std::vector<std::vector<double>> transposed_data(this->cols, std::vector<double>(this->rows, 0));
    for (int i = 0; i < this->rows; i++)
        for (int j = 0; j < this->cols; j++)
            transposed_data[j][i] = this->data[i][j];
    return {this->cols, this->rows, transposed_data};
}

void Matrix::randomize() {
    std::default_random_engine gen(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> dist(-1, 1);
    for (auto &row : this->data)
        for (auto &col : row)
            col = dist(gen);
}

Matrix Matrix::log() const {
    auto result = std::vector<std::vector<double>>(this->rows, std::vector<double>(this->cols, 0));
    for (int i = 0; i < this->rows; i++)
        for (int j = 0; j < this->cols; j++)
            result[i][j] = std::log(this->data[i][j]);
    return {this->rows, this->cols, result};
}

void Matrix::set_value(uint32_t row, uint32_t col, double value) {
    this->data[row][col] = value;
}

void Matrix::set_row(uint32_t row, const std::vector<double> &values) {
    this->data[row] = values;
}

void Matrix::set_col(uint32_t col, const std::vector<double> &values) {
    for (int i = 0; i < this->rows; i++)
        this->data[i][col] = values[i];
}

void Matrix::set_values(const std::vector<std::vector<double>> &values) {
    this->data = values;
}

void Matrix::add_row(const std::vector<double> &values) {
    this->data.push_back(values);
    this->rows++;
}

void Matrix::add_col(const std::vector<double> &values) {
    for (int i = 0; i < this->rows; i++)
        this->data[i].push_back(values[i]);
    this->cols++;
}

double Matrix::get_value(uint32_t row, uint32_t col) const {
    return this->data[row][col];
}

Matrix Matrix::get_row(uint32_t row) const {
    auto row_data = this->data[row];
    auto new_matrix_data = std::vector<std::vector<double>>(1, row_data);
    Matrix result(1, this->cols, new_matrix_data);
    return result;
}

Matrix Matrix::get_col(uint32_t col) const {
    std::vector<double> col_data;
    col_data.reserve(this->rows);
    for (int i = 0; i < this->rows; i++)
        col_data.push_back(this->data[i][col]);
    auto new_matrix_data = std::vector<std::vector<double>>(this->rows, col_data);
    Matrix result(this->rows, 1, new_matrix_data);
    return result;
}

std::vector<std::vector<double>> Matrix::get_values() const {
    return this->data;
}

std::vector<uint32_t> Matrix::get_dims() const {
    return {this->rows, this->cols};
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

Matrix Matrix::operator+(const Matrix &other) const {
    auto result = std::vector<std::vector<double>>(this->rows, std::vector<double>(this->cols, 0));
    for (int i = 0; i < this->rows; i++)
        for (int j = 0; j < this->cols; j++)
            result[i][j] = this->data[i][j] + other.data[i][j];
    return {this->rows, this->cols, result};
}

Matrix Matrix::operator-(const Matrix &other) const {
    auto result = std::vector<std::vector<double>>(this->rows, std::vector<double>(this->cols, 0));
    for (int i = 0; i < this->rows; i++)
        for (int j = 0; j < this->cols; j++)
            result[i][j] = this->data[i][j] - other.data[i][j];
    return {this->rows, this->cols, result};
}

Matrix Matrix::operator*(const Matrix &other) const {
    if (this->cols != other.rows)
        throw std::runtime_error("Matrix multiplication error: incompatible dimensions");

    auto result = std::vector<std::vector<double>>(this->rows, std::vector<double>(other.cols, 0));
    for (int i = 0; i < this->rows; i++)
        for (int j = 0; j < other.cols; j++)
            for (int k = 0; k < this->cols; k++)
                result[i][j] += this->data[i][k] * other.data[k][j];

    return {this->rows, other.cols, result};
}

std::ostream &operator<<(std::ostream &os, const Matrix &matrix) {
    for (const auto &row : matrix.data) {
        for (const auto &col : row)
            os << col << " ";
        os << std::endl;
    }
    return os;
}
