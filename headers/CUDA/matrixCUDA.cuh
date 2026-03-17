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
class matrix<CUDA> 
{
private:
    float* data;
    size_t r,c, h;
    size_t n;

    bool owns_memory = true;
    
public:

    static constexpr int THREADS_1D = 256;
    static constexpr int THREADS_2D = 16;


    /** @brief creates empty matrix of shape 0x0x0 */
    matrix();

    /** @brief copy constructor, which does a deepcopy. */
    matrix<CUDA>(const matrix<CUDA>& other);

    /** @brief move constructor */
    matrix<CUDA>(matrix<CUDA>&& other) noexcept;

    /** @brief creates a matrix of shape rows x columns x 1 (non stacked)*/
    matrix<CUDA>(const size_t rows, const size_t columns);

    /** 
     * @brief creates a matrix of shape rows x columns x 1 (non stacked)
     * 
     * @param values Values for the matrix. values.size() needs to be equal to rows * columns.
     * */
    matrix<CUDA>(const size_t rows, const size_t columns, const std::vector<float>& values);

    /** 
     * @brief creates a matrix of shape rows x columns x 1 (non stacked)
     * 
     * @param value set all values of the matrix to this value.
     * */
    matrix<CUDA>(const size_t rows, const size_t columns, float val);

    /** 
     * @brief creates a matrix of shape rows x columns x 1 (non stacked), and initalises it with random values from start to end.
     * 
     * @param start lowest value
     * @param end  highes value
     * */
    matrix<CUDA>(const size_t rows, const size_t columns, float start, float end);

    /** 
     * @brief Creates a view of a matrix with the shape rows x columns x height (stacked)
     * 
     * @param ptr pointer to the raw data of matrix.
     * 
     * @note Use .slice_matrix(...) instead.
     * */
    matrix<CUDA>(float* ptr, const size_t rows, const size_t columns, const size_t height);

    ~matrix<CUDA>();

    /**
     * @brief Creates a empty stacked matrix of shape rows x columns x height.
     * 
     */
    static matrix<CUDA> create_stacked_matrix(const size_t rows, const size_t columns, const size_t height);

    /**
     * @brief Creates a stacked matrix of shape rows x columns x height.
     * 
     * @param values data, values.shape() needs to be equal to rows * columns * height
     */
    static matrix<CUDA> create_stacked_matrix(const size_t rows, const size_t columns, const size_t height, const std::vector<float>& values);

    /**
     * @brief Creates a stacked matrix of shape rows x columns x height.
     * 
     * @param value Sets all values to this value.
     */
    static matrix<CUDA> create_stacked_matrix(const size_t rows, const size_t columns, const size_t height, float val);

    /** 
     * @brief creates a stacked matrix of shape rows x columns x height, and initalises it with random values from start to end.
     * 
     * @param start lowest value
     * @param end  highes value
     * */
    static matrix<CUDA> create_stacked_matrix(const size_t rows, const size_t columns, const size_t height, float start, float end);


    /**
     * @brief Creates a view for a stacked matrix.
     * 
     * @param start start height (0 to .height()) 
     * @param end end height (exclusive)
     */
    matrix<CUDA> slice_stacked_matrix(size_t start, size_t end);



    /**
     * @brief Sums all stacks of a matrix together.
     */    
    static matrix<CUDA> reduce_sum(const matrix<CUDA> &a);


    /**
     * @brief Broadcasts the mat addition.
     * @param a Matrix which will recieve the broadcast (needs to have equal rows & cols to b)
     * @param b Matrix which will be broadcasted (needs to have height of 1)
     */
    static matrix<CUDA> bcast_add_to_stacked_matrix(const matrix<CUDA>& a, const matrix<CUDA>& b);

    /**
     * @brief Broadcasts the hadamard product.
     * @param a Matrix which will recieve the broadcast (needs to have equal rows & cols to b)
     * @param b Matrix which will be broadcasted (needs to have height of 1)
     */
    static matrix<CUDA> bcast_hadamard_to_stacked_matrix(const matrix<CUDA>& a, const matrix<CUDA>& b);

    /**
     * @brief Broadcasts reversed matrix multiplication.
     * @param a Matrix which will recieve the broadcast (needs to have equal rows & cols to b)
     * @param b Matrix which will be broadcasted (needs to have height of 1)
     * 
     * => Calculates B x A, where A broadcasts to B.
     */
    static matrix<CUDA> bcast_reversed_mat_mul_to_stacked_matrix(const matrix<CUDA>& a, const matrix<CUDA>& b);

    /**
     * @brief Broadcasts matrix multiplication.
     * @param a Matrix which will recieve the broadcast (needs to have equal rows & cols to b)
     * @param b Matrix which will be broadcasted (needs to have height of 1)
     * 
     * => Calculates A x B
     */
    static matrix<CUDA> bcast_mat_mul_to_stacked_matrix(const matrix<CUDA>& a, const matrix<CUDA>& b);

    /**
     * @brief Broadcasts scaling .
     * @param a Matrix which will recieve the broadcast (needs to have equal height to b)
     * @param b Matrix which will be broadcasted (needs to be of shape 1x1xh)
     * 
     * => Calculates A x B
     */
    static matrix<CUDA> bcast_scale_to_stacked_matrix(const matrix<CUDA>& a, const matrix<CUDA>& b);


    float operator[](size_t index) const;

    /**
     * @brief sets a value at index to val
     * @param index Index of the entire matrix from 0 to rows * cols * height.
     */
    void set(size_t index, float val);

    matrix<CUDA>& operator=(const matrix<CUDA>& other);
    matrix<CUDA>& operator=(matrix<CUDA>&& other) noexcept;

    matrix<CUDA> operator%(const matrix<CUDA> &a) const;
    matrix<CUDA> operator+(const matrix<CUDA> &a) const;
    matrix<CUDA> operator+(const float &a) const;
    matrix<CUDA> operator-(const matrix<CUDA> &a) const;
    matrix<CUDA> operator-(const float &a) const;
    matrix<CUDA> operator*(const float &a) const;
    matrix<CUDA> operator*(const matrix<CUDA> &a) const;
    matrix<CUDA> operator+=(const matrix<CUDA> &a);
    matrix<CUDA> operator-=(const matrix<CUDA> &a);

    /**
     * @brief Transposed the matrix
     */
    static matrix<CUDA> transpose(const matrix<CUDA> &a);


     

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
    matrix<CUDA> min() const;
    /** @brief max elements of each matrix. @note The method doesnt change the input matrix.  */
    matrix<CUDA> max() const;
    /** @brief sum all elements inside a  matrix. @note The method doesnt change the input matrix.  */
    matrix<CUDA> sum() const; 
    /** @brief L2 of all elements inside a matrix. @note The method doesnt change the input matrix.  */
    matrix<CUDA> L2() const;
    
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

    static matrix<CUDA> sqrt(const matrix<CUDA> &a);
    static matrix<CUDA> square(const matrix<CUDA> &a);
    static matrix<CUDA> reciprocal(const matrix<CUDA> &a);
    static matrix<CUDA> exp(const matrix<CUDA> &a);
    static matrix<CUDA> log2(const matrix<CUDA> &a);



};

matrix<CUDA> operator*(float val, const matrix<CUDA>& a);
matrix<CUDA> operator+(float val, const matrix<CUDA>& a);
matrix<CUDA> operator-(float val, const matrix<CUDA>& a);


template<>
class memory_pool<CUDA>
{
private:
    std::unordered_map<size_t, std::vector<float*>> free_blocks;

public:
    static memory_pool<CUDA>& instance();
    float* allocate(size_t n);
    void deallocate(float* ptr, size_t n);
    ~memory_pool<CUDA>();
};
