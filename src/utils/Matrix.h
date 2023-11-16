#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <chrono>

/**
 * Class representing a matrix of doubles
 * Main reason for this class is to make matrix operations easier
 */
class Matrix {
private:
    /** Number of rows */
    uint32_t rows;
    /** Number of columns */
    uint32_t cols;
    /** Data of the matrix */
    std::vector<std::vector<double>> data;

public:
    /**
     * Default constructor
     * @param rows Number of rows
     * @param cols Number of columns
     * @param randomize Flag if the matrix should be randomized
     */
    Matrix(uint32_t rows, uint32_t cols, bool randomize = false);
    /**
     * Constructor with data
     * @param rows Number of rows
     * @param cols Number of columns
     * @param data Data of the matrix
     */
    Matrix(uint32_t rows, uint32_t cols, const std::vector<std::vector<double>> &data);
    /**
     * Copy constructor
     * @param other Matrix to copy
     */
    Matrix(const Matrix &other) noexcept;
    /**
     * Move constructor
     * @param other Matrix to move
     */
    Matrix(Matrix &&other) noexcept;
    /**
     * Default destructor
     */
    ~Matrix();

    /**
     * Transpose the matrix
     * @return New transposed matrix (original matrix is not changed)
     */
    [[nodiscard]] Matrix transpose() const;
    /**
     * Randomize the matrix data values (uniform real distribution from -1 to 1)
     */
    void randomize();
    /**
     * Apply log function to all values in the matrix
     * @return New matrix with log values (original matrix is not changed)
     */
    [[nodiscard]] Matrix log() const;

    /**
     * Set the value at the given position
     * @param row Row index
     * @param col Column index
     * @param value Value to set
     */
    void set_value(uint32_t row, uint32_t col, double value);
    /**
     * Set the values of the given row
     * @param row Row index
     * @param values Values to set
     */
    void set_row(uint32_t row, const std::vector<double> &values);
    /**
     * Set the values of the given column
     * @param col Column index
     * @param values Values to set
     */
    void set_col(uint32_t col, const std::vector<double> &values);
    /**
     * Set the values of the matrix
     * @param values Values to set
     */
    void set_values(const std::vector<std::vector<double>> &values);
    /**
     * Add a row to the matrix
     * @param values Values of the row to add
     */
    void add_row(const std::vector<double> &values);
    /**
     * Add a column to the matrix
     * @param values Values of the column to add
     */
    void add_col(const std::vector<double> &values);
    /**
     * Remove a row from the matrix
     * @param row_idx Index of the row to remove
     */
    void remove_row(uint32_t row_idx);
    /**
     * Remove a column from the matrix
     * @param col_idx Index of the column to remove
     */
    void remove_col(uint32_t col_idx);
    /**
     * Get the value at the given position
     * @param row Row index
     * @param col Column index
     * @return Value at the given position
     */
    [[nodiscard]] double get_value(uint32_t row, uint32_t col) const;
    /**
     * Get the row at the given position
     * @param row Row index
     * @return Row at the given position
     */
    [[nodiscard]] Matrix get_row(uint32_t row) const;
    /**
     * Get the column at the given position
     * @param col Column index
     * @return Column at the given position
     */
    [[nodiscard]] Matrix get_col(uint32_t col) const;
    /**
     * Get the values of the matrix
     * @return Values of the matrix
     */
    [[nodiscard]] std::vector<std::vector<double>> get_values() const;
    /**
     * Get the dimensions of the matrix (rows x cols)
     * @return Dimensions of the matrix (rows x cols)
     */
    [[nodiscard]] std::vector<uint32_t> get_dims() const;
    /**
     * Get the index of the maximum value in the matrix
     * @return Index of the maximum value in the matrix
     */
    [[nodiscard]] uint32_t argmax() const;

    /**
     * Copy assignment operator
     * @param other Matrix to copy
     * @return Copied matrix
     */
    Matrix &operator=(const Matrix &other) noexcept;
    /**
     * Move assignment operator
     * @param other Matrix to move
     * @return Moved matrix
     */
    Matrix &operator=(Matrix &&other) noexcept;
    /**
     * Overloaded addition operator
     * @param other Matrix to add
     * @return Result of the addition
     */
    Matrix operator+(const Matrix &other) const;
    /**
     * Overloaded subtraction operator
     * @param other Matrix to subtract
     * @return Result of the subtraction
     */
    Matrix operator-(const Matrix &other) const;
    /**
     * Overloaded multiplication operator
     * @param other Matrix to multiply
     * @return Result of the matrix multiplication
     */
    Matrix operator*(const Matrix &other) const;
    /**
     * Overloaded multiplication operator
     * @param scalar Scalar to multiply
     * @return Result of the scalar multiplication
     */
    Matrix operator*(double scalar) const;
    /**
     * Overloaded left shift operator (for printing)
     * @param os Output stream
     * @param matrix Matrix to print (this)
     * @return Output stream
     */
    friend std::ostream &operator<<(std::ostream &os, const Matrix &matrix);
};
