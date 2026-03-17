#pragma once
#include "matrix.h"
#include <vector>
#include <iostream>
#include <numeric>
#include <unordered_map>


/**
 * @brief Matrix class, which enables to user to do linear algebra operations. 
 */
template<>
class matrix<CPU> 
{
private:
    float* data;
    size_t r,c, h;
    size_t n;
    bool owns_memory = true;
    
public:

    /** @brief creates empty matrix of shape 0x0x0 */
    matrix();

    /** @brief copy constructor, which does a deepcopy. */
    matrix<CPU>(const matrix<CPU>& other);

    /** @brief move constructor */
    matrix<CPU>(matrix<CPU>&& other) noexcept;

    /** @brief creates a matrix of shape rows x columns x 1 (non stacked)*/
    matrix<CPU>(const size_t rows, const size_t columns);

    /** 
     * @brief creates a matrix of shape rows x columns x 1 (non stacked)
     * 
     * @param values Values for the matrix. values.size() needs to be equal to rows * columns.
     * */
    matrix<CPU>(const size_t rows, const size_t columns, const std::vector<float>& values);

    /** 
     * @brief creates a matrix of shape rows x columns x 1 (non stacked)
     * 
     * @param value set all values of the matrix to this value.
     * */
    matrix<CPU>(const size_t rows, const size_t columns, float val);

    /** 
     * @brief creates a matrix of shape rows x columns x 1 (non stacked), and initalises it with random values from start to end.
     * 
     * @param start lowest value
     * @param end  highes value
     * */
    matrix<CPU>(const size_t rows, const size_t columns, float start, float end);

    /** 
     * @brief Creates a view of a matrix with the shape rows x columns x height (stacked)
     * 
     * @param ptr pointer to the raw data of matrix.
     * 
     * @note Use .slice_matrix(...) instead.
     * */
    matrix<CPU>(float* ptr, const size_t rows, const size_t columns, const size_t height);
    ~matrix<CPU>();

    /**
     * @brief Creates a empty stacked matrix of shape rows x columns x height.
     * 
     */
    static matrix<CPU> create_stacked_matrix(const size_t rows, const size_t columns, const size_t height);

    /**
     * @brief Creates a stacked matrix of shape rows x columns x height.
     * 
     * @param values data, values.shape() needs to be equal to rows * columns * height
     */
    static matrix<CPU> create_stacked_matrix(const size_t rows, const size_t columns, const size_t height, const std::vector<float>& values);

    /**
     * @brief Creates a stacked matrix of shape rows x columns x height.
     * 
     * @param value Sets all values to this value.
     */
    static matrix<CPU> create_stacked_matrix(const size_t rows, const size_t columns, const size_t height, float val);

    /** 
     * @brief creates a stacked matrix of shape rows x columns x height, and initalises it with random values from start to end.
     * 
     * @param start lowest value
     * @param end  highes value
     * */
    static matrix<CPU> create_stacked_matrix(const size_t rows, const size_t columns, const size_t height, float start, float end);

     /**
     * @brief Creates a view for a stacked matrix.
     * 
     * @param start start height (0 to .height()) 
     * @param end end height (exclusive)
     */
    matrix<CPU> slice_stacked_matrix(size_t start, size_t end);

    /**
     * @brief Sums all stacks of a matrix together.
     */    
    static matrix<CPU> reduce_sum(const matrix<CPU> &a);

    /**
     * @brief Broadcasts the mat addition.
     * @param a Matrix which will recieve the broadcast (needs to have equal rows & cols to b)
     * @param b Matrix which will be broadcasted (needs to have height of 1)
     */
    static matrix<CPU> bcast_add_to_stacked_matrix(const matrix<CPU>& a, const matrix<CPU>& b);

    /**
     * @brief Broadcasts the hadamard product.
     * @param a Matrix which will recieve the broadcast (needs to have equal rows & cols to b)
     * @param b Matrix which will be broadcasted (needs to have height of 1)
     */
    static matrix<CPU> bcast_hadamard_to_stacked_matrix(const matrix<CPU>& a, const matrix<CPU>& b);

    /**
     * @brief Broadcasts reversed matrix multiplication.
     * @param a Matrix which will recieve the broadcast (needs to have equal rows & cols to b)
     * @param b Matrix which will be broadcasted (needs to have height of 1)
     * 
     * => Calculates B x A, where A broadcasts to B.
     */
    static matrix<CPU> bcast_reversed_mat_mul_to_stacked_matrix(const matrix<CPU>& a, const matrix<CPU>& b);

    /**
     * @brief Broadcasts matrix multiplication.
     * @param a Matrix which will recieve the broadcast (needs to have equal rows & cols to b)
     * @param b Matrix which will be broadcasted (needs to have height of 1)
     * 
     * => Calculates A x B
     */
    static matrix<CPU> bcast_mat_mul_to_stacked_matrix(const matrix<CPU>& a, const matrix<CPU>& b);

    /**
     * @brief Broadcasts scaling .
     * @param a Matrix which will recieve the broadcast (needs to have equal height to b)
     * @param b Matrix which will be broadcasted (needs to be of shape 1x1xh)
     * 
     * => Calculates A x B
     */
    static matrix<CPU> bcast_scale_to_stacked_matrix(const matrix<CPU>& a, const matrix<CPU>& b);


    const float operator[](size_t index) const;
    float& operator[](size_t index);

    /**
     * @brief sets a value at index to val
     * @param index Index of the entire matrix from 0 to rows * cols * height.
     */
    void set(size_t index, float val);

    matrix<CPU>& operator=(const matrix<CPU>& other);
    matrix<CPU>& operator=(matrix<CPU>&& other) noexcept;

    matrix<CPU> operator%(const matrix<CPU> &a) const;
    matrix<CPU> operator+(const matrix<CPU> &a) const;
    matrix<CPU> operator+(const float &a) const;
    matrix<CPU> operator-(const matrix<CPU> &a) const;
    matrix<CPU> operator-(const float &a) const;
    matrix<CPU> operator*(const float &a) const;
    matrix<CPU> operator*(const matrix<CPU> &a) const;
    matrix<CPU> operator+=(const matrix<CPU> &a);
    matrix<CPU> operator-=(const matrix<CPU> &a);

    /**
     * @brief Transposed the matrix
     */
    static matrix<CPU> transpose(const matrix<CPU> &a);


     
    size_t rows() const;
    size_t columns() const;
    size_t height() const;

    /** @return rows * columns * height*/
    size_t elements() const;

    /** @return rows * colums */
    size_t mat_elements() const;

    bool empty() const;
    float* raw();
    float* raw() const;
    std::vector<float> values();

    /** @brief min elements of each matrix. @note The method doesnt change the input matrix. */
    matrix<CPU> min() const;
    /** @brief max elements of each matrix. @note The method doesnt change the input matrix.  */
    matrix<CPU> max() const;
    /** @brief sum all elements inside a  matrix. @note The method doesnt change the input matrix.  */
    matrix<CPU> sum() const; 
    
    /** @brief L2 of all elements inside a matrix. @note The method doesnt change the input matrix.  */
    matrix<CPU> L2() const;
    
    /** 
     * @brief argmax of all elements inside a matrix 
     * @return a vector with a global index of each argmax
     * @note global index = [0, rows * columns * height]. The Operator [] and .set(...) also use these.
    */
    std::vector<size_t> argmax() const;

    /** 
     * @brief argmin of all elements inside a matrix 
     * @return a vector with a global index of each argmax
     * @note global index = [0, rows * columns * height]. The Operator [] and .set(index, val) also use these.
    */
    std::vector<size_t> argmin() const;


    void print() const;
    void print_shape() const;

    /** 
     * @brief Sets all elements to value.
    */
    void set(float val);

    static matrix<CPU> sqrt(const matrix<CPU> &a);
    static matrix<CPU> square(const matrix<CPU> &a);
    static matrix<CPU> reciprocal(const matrix<CPU> &a);
    static matrix<CPU> exp(const matrix<CPU> &a);
    static matrix<CPU> log2(const matrix<CPU> &a);

};

matrix<CPU> operator*(float val, const matrix<CPU>& a);
matrix<CPU> operator+(float val, const matrix<CPU>& a);
matrix<CPU> operator-(float val, const matrix<CPU>& a);


template<>
class memory_pool<CPU>
{
private:
    std::unordered_map<size_t, std::vector<float*>> free_blocks;

public:
    static memory_pool<CPU>& instance();
    float* allocate(size_t n);
    void deallocate(float* ptr, size_t n);
    ~memory_pool<CPU>();
};
