#pragma once


__global__ void matrix_kernel_sum(const float* data, float *result, const size_t mat_size);

__global__ void matrix_kernel_argmax(const float* data, size_t* result, const size_t mat_size);

__global__ void matrix_kernel_argmin(const float* data, size_t* result, const size_t mat_size);

__global__ void matrix_kernel_sqrt(float* data, size_t n);

__global__ void matrix_kernel_square(float* data, size_t n);

__global__ void matrix_kernel_reciprocal(float* data, size_t n);

__global__ void matrix_kernel_hadamard(const float *A, const float *B, float *result, const size_t n);

__global__ void matrix_kernel_add(const float *A, const float *B, float *result, const size_t n);

__global__ void matrix_kernel_sub(const float *A, const float *B, float *result, const size_t n);

__global__ void matrix_kernel_scale(const float *A, float *result, const float value, const size_t n);

__global__ void matrix_kernel_add_value(const float *A, float *result, const float value, const size_t n);

__global__ void matrix_kernel_mat_mul(const float *A, const float *B, float *result, const size_t result_rows, const size_t result_cols, const size_t length);

__global__ void matrix_kernel_add_mat_to_stacked_matrix(const float* A, const float* B, float *result, const size_t mat_size, const size_t n);

__global__ void matrix_kernel_transpose(const float* A, float* result, const size_t result_rows, const size_t result_columns, const size_t n);







