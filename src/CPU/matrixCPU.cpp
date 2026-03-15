#include "matrixCPU.h"
#include <random>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cstring>


matrix<CPU>::matrix() : n(0), h(0), c(0), r(0), data(nullptr), owns_memory(false)
{
}

matrix<CPU>::matrix(const matrix<CPU>& other) : r(other.r), c(other.c), n(other.n), h(other.h)
{
    this->data = memory_pool<CPU>::instance().allocate(n);
    memcpy(this->data, other.data, sizeof(float) * other.n);
}

matrix<CPU>::matrix(matrix<CPU>&& other) noexcept : n(other.n), r(other.r), c(other.c), h(other.h), data(other.data), owns_memory(other.owns_memory)
{
    other.n = 0;
    other.r = 0;
    other.c = 0;
    other.h = 0;
    other.data = nullptr;
}

matrix<CPU>::matrix(const size_t rows, const size_t columns) : r(rows), c(columns), h(1)
{
    this->n = r * c;
    this->data = memory_pool<CPU>::instance().allocate(n);
}

matrix<CPU>::matrix(const size_t rows, const size_t columns, const std::vector<float>& values) : matrix(rows, columns)
{

    if(values.size() != n)
    {
        throw std::runtime_error("Matrix : Size of matrix does not equal the input vector size");
    }
    memcpy(this->data, values.data(), sizeof(float) * n);
}

matrix<CPU>::matrix(const size_t rows, const size_t columns, float val) : matrix(rows, columns)
{
    std::vector<float> arr(n, val);
    memcpy(this->data, arr.data(), sizeof(float) * n);
}

matrix<CPU>::matrix(const size_t rows, const size_t columns, float start, float end) : matrix(rows, columns)
{
    std::vector<float> arr(n);
    std::mt19937 gen(
        std::random_device{}()
    );
    std::uniform_real_distribution<float> dist(start, end);

    for (size_t i = 0; i < this->n; ++i){
        arr[i] = dist(gen);
    }
    memcpy(this->data, arr.data(), sizeof(float) * n);

}

matrix<CPU>::matrix(float *ptr, const size_t rows, const size_t columns, const size_t height) : data(ptr), r(rows), c(columns), h(height), owns_memory(false)
{
    this->n = rows * columns * height;
}

matrix<CPU>::~matrix()
{
    if(owns_memory)
        memory_pool<CPU>::instance().deallocate(data, n);
}



matrix<CPU> matrix<CPU>::create_stacked_matrix(const size_t rows, const size_t columns, const size_t height)
{
    matrix<CPU> result;
    result.r = rows;
    result.c = columns;
    result.h = height;
    result.n = rows * columns * height;
    result.owns_memory = true;
    result.data = memory_pool<CPU>::instance().allocate(result.n);
    return result;
}

matrix<CPU> matrix<CPU>::create_stacked_matrix(const size_t rows, const size_t columns, const size_t height, const std::vector<float>& values)
{
    if(values.size() != rows * columns * height)
        throw std::runtime_error("create_stacked_matrix : vector size and mat dim's dont match.");

    matrix<CPU> result = create_stacked_matrix(rows, columns, height);
    memcpy(result.data, values.data(), result.n * sizeof(float));
    return result;
}

matrix<CPU> matrix<CPU>::create_stacked_matrix(const size_t rows, const size_t columns, const size_t height, float val)
{   

    matrix<CPU> result = create_stacked_matrix(rows, columns, height);
    std::vector<float> arr(result.n, val);

    memcpy(result.data, arr.data(), result.n * sizeof(float));
    return result;
}

matrix<CPU> matrix<CPU>::create_stacked_matrix(const size_t rows, const size_t columns, const size_t height, float start, float end)
{
    matrix<CPU> result = create_stacked_matrix(rows, columns, height);
    std::vector<float> arr(result.n);
    
    std::mt19937 gen(
        std::random_device{}()
    );
    std::uniform_real_distribution<float> dist(start, end);

    for (size_t i = 0; i < result.n; ++i)
    {
        arr[i] = dist(gen);
    }
    memcpy(result.data, arr.data(), sizeof(float) * result.n);
    return result;
}



matrix<CPU> matrix<CPU>::slice_stacked_matrix(size_t start, size_t end)
{   
    if(this->h < end)
        throw std::runtime_error("slice_stacked_matrix : matrix height to small.");

    return matrix<CPU>(this->data + start * mat_elements(), r,c, end - start);
}

// ------------------------------------------

matrix<CPU> matrix<CPU>::reduce_sum(const matrix<CPU> &a)
{
    matrix<CPU> res = create_stacked_matrix(a.rows(),a.columns(), 1);
    
    for(size_t i = 0; i < a.mat_elements(); i++)
    {
        
        float sum = 0;
        for(size_t layer = 0; layer < a.height(); layer++ )
        {
            sum += a[layer * a.mat_elements() + i];
        }
        res[i] = sum;
    }

    return res;
}

matrix<CPU> matrix<CPU>::bcast_add_to_stacked_matrix(const matrix<CPU> &a, const matrix<CPU> &b)
{   

    size_t mat_size = a.mat_elements();
    
    if(mat_size != b.mat_elements() || b.height() != 1)
        throw std::runtime_error("bcast_add_to_stacked_matrix : matrix dimensions must be equal and the height of the second matrix needs to be 1.");   

    matrix<CPU> res = create_stacked_matrix(a.rows(), a.columns(), a.height());

    for(size_t layer = 0; layer < a.height(); layer++)
    {
        const size_t offset = layer * mat_size;
        for(size_t i = 0; i < mat_size; i++ )
            res[offset + i] = a[offset + i] + b[i];
    }
    return res;

}

matrix<CPU> matrix<CPU>::bcast_hadamard_to_stacked_matrix(const matrix<CPU> &a, const matrix<CPU> &b)
{
    size_t mat_size = a.mat_elements();
    if(b.height() != 1 || a.rows() != b.rows() || a.columns() != b.columns() )
        throw std::runtime_error("bcast_hadamard_to_stacked_matrix : height needs to be equal to 1");    

    matrix<CPU> res = create_stacked_matrix(a.rows(), a.columns(), a.height());

    for(size_t layer = 0; layer < a.height(); layer++)
    {
        const size_t offset = layer * mat_size;
        for(size_t i = 0; i < mat_size; i++ )
            res[offset + i] = a[offset + i] * b[i];
    }
    return res;
    
}

matrix<CPU> matrix<CPU>::bcast_reversed_mat_mul_to_stacked_matrix(const matrix<CPU> &a, const matrix<CPU> &b)
{

    
    if(b.columns() != a.rows() || b.height() != 1)
        throw std::runtime_error("bcast_reversed_mat_mul_to_stacked_matrix : matrix shapes do not match. b.columns() needs to be equal to a.rows() (reversed mat mul!) ");

    
    const size_t rows = b.rows();
    const size_t cols = a.columns();
    const size_t inner = b.columns();
    const size_t height = a.height();

    matrix<CPU> res = create_stacked_matrix(rows, cols, height);


    for(size_t layer = 0; layer < height; layer++)
        for(size_t row = 0; row < rows; row++) 
        {   
            const size_t a_offset = (a.rows() * cols) * layer;
            const size_t res_offset = (rows * cols) * layer;

            for(size_t col = 0; col < cols; col++)
            {
                float sum = 0;
                for(size_t i = 0; i < inner; i++)
                    sum += b[row * inner + i] * a[a_offset + i * cols + col];
                res[res_offset + row * cols + col] = sum;
            }
        }

    return res;
}

matrix<CPU> matrix<CPU>::bcast_mat_mul_to_stacked_matrix(const matrix<CPU> &a, const matrix<CPU> &b)
{
    if(a.columns() != b.rows() || b.height() != 1)
        throw std::runtime_error("bcast_mat_mul_to_stacked_matrix : matrix shapes do not match. a.columns() needs to be equal to b.rows()");

   

    const size_t rows = a.rows();
    const size_t cols = b.columns();
    const size_t inner = a.columns();
    const size_t height = a.height();

    matrix<CPU> res = create_stacked_matrix(rows, cols, height);

    
    for(size_t layer = 0; layer < height; layer++)
        for(size_t row = 0; row < rows; row++) 
        {   
            const size_t a_offset = (rows * inner) * layer;
            const size_t res_offset = (rows * cols) * layer;

            for(size_t col = 0; col < cols; col++)
            {
                float sum = 0;
                for(size_t i = 0; i < inner; i++)
                    sum += a[a_offset + row * inner + i] * b[i * cols + col];
                res[res_offset + row * cols + col] = sum;
            }
        }

    return res;


}

matrix<CPU> matrix<CPU>::bcast_scale_to_stacked_matrix(const matrix<CPU> &a, const matrix<CPU> &b)
{
    size_t mat_size = a.mat_elements();
    if(b.rows() != 1 || b.columns() != 1 || b.height() != a.height() )
        throw std::runtime_error("bcast_scale_to_stacked_matrix : Height of matrices need to match and second matrix has to be of shape 1x1xh");    

    matrix<CPU> res = create_stacked_matrix(a.rows(), a.columns(), a.height());

    for(size_t layer = 0; layer < a.height(); layer++)
    {
        const size_t offset = layer * mat_size;
        for(size_t i = 0; i < mat_size; i++ )
            res[offset + i] = a[offset + i] * b[layer];
    }
    return res;
    
}

size_t matrix<CPU>::rows() const
{
    return r;
}


size_t  matrix<CPU>::columns() const
{
    return c;
}

size_t  matrix<CPU>::height() const
{
    return h;
}

size_t  matrix<CPU>::elements() const
{
    return n;
}

size_t matrix<CPU>::mat_elements() const
{
    return this->r * this->c;
}

bool  matrix<CPU>::empty() const
{
    return (this->n==0);
}

float* matrix<CPU>::raw()
{
    return data;
}

float *matrix<CPU>::raw() const
{
    return data;
}

std::vector<float> matrix<CPU>::values()
{
    std::vector<float> arr(this->n);
    memcpy(arr.data(), this->data, sizeof(float) * this->n);
    return arr;
}

matrix<CPU> matrix<CPU>::sum() const
{
    matrix<CPU> res = create_stacked_matrix(1,1, this->h);
    for(size_t layer = 0; layer < this->h; layer++ )
    {
        auto begin = data + mat_elements() * layer;
        auto end = data + mat_elements() * (layer+1);
        res[layer] = std::accumulate(begin, end, 0.0f);
    }
    return res;
}

matrix<CPU> matrix<CPU>::L2() const
{
    matrix<CPU> squared = matrix<CPU>::square(*this);
    return squared.sum();
}

matrix<CPU> matrix<CPU>::max() const
{
    matrix<CPU> res = create_stacked_matrix(1,1, this->h);
    for(size_t layer = 0; layer < this->h; layer++ )
    {
        auto begin = data + mat_elements() * layer;
        auto end = data + mat_elements() * (layer+1);
        res[layer] = *std::max_element(begin, end);
    }
    return res;
}

matrix<CPU> matrix<CPU>::min() const
{
    matrix<CPU> res = create_stacked_matrix(1,1, this->h);
    for(size_t layer = 0; layer < this->h; layer++ )
    {
        auto begin = data + mat_elements() * layer;
        auto end = data + mat_elements() * (layer+1);
        res[layer] = *std::min_element(begin, end);
    }
    return res;
}





std::vector<size_t>  matrix<CPU>::argmax() const
{
    std::vector<size_t> res(this->h);
    for(size_t layer = 0; layer < this->h; layer++ )
    {
        auto begin = data + mat_elements() * layer;
        auto end = data + mat_elements() * (layer+1);
        auto max = std::max_element(begin, end);
        res[layer] = std::distance(data, max);
    }
    return res;
}

std::vector<size_t>   matrix<CPU>::argmin() const
{
    std::vector<size_t> res(this->h);
    for(size_t layer = 0; layer < this->h; layer++ )
    {
        auto begin = data + mat_elements() * layer;
        auto end = data + mat_elements() * (layer+1);
        auto min = std::min_element(begin, end);
        res[layer] = std::distance(data, min);
    }
    return res;

}


void  matrix<CPU>::print() const 
{

    for(int l = 0; l < this->h; l++)
    {
        std::cout << "----------- Matrix : " << l << " -----------" << std::endl;
        size_t offset = l * mat_elements();
        for(size_t row = 0; row < this->r; row++)
        {
            for(size_t col = 0; col< this->c; col++)
            {
                std::cout << data[row * this->c + col + offset] << " ";
            }
            std::cout << std::endl;
        }
    }

}

void  matrix<CPU>::print_shape() const
{
    std::cout << r << " " << c << " " << h << std::endl;
}

void  matrix<CPU>::set(float val)
{

    if(val == 0.0f)
    {
        memset(this->data, 0, n * sizeof(float));  
        return;
    }

    for(size_t i = 0; i < this->n; i++)
        data[i] = val;

}

matrix<CPU>  matrix<CPU>::sqrt(const matrix<CPU> &a)
{
    
    matrix<CPU> res = create_stacked_matrix(a.rows(), a.columns(), a.height());
    for(size_t i = 0; i < a.elements(); i++)
        res[i] = std::sqrt(a[i]);
    return res;

}

matrix<CPU>  matrix<CPU>::square(const matrix<CPU> &a)
{
    matrix<CPU> res = create_stacked_matrix(a.rows(), a.columns(), a.height());
    for(size_t i = 0; i < a.elements(); i++)
        res[i] = a[i] * a[i];
    return res;

}

matrix<CPU>  matrix<CPU>::reciprocal(const matrix<CPU> &a)
{
    matrix<CPU> res = create_stacked_matrix(a.rows(), a.columns(), a.height());
    for(size_t i = 0; i < a.elements(); i++)
        res[i] = 1/a[i];
    return res;

}

matrix<CPU> matrix<CPU>::exp(const matrix<CPU> &a)
{
    matrix<CPU> res = create_stacked_matrix(a.rows(), a.columns(), a.height());
    for(size_t i = 0; i < a.elements(); i++)
        res[i] = std::exp(a[i]);
    return res;

}

matrix<CPU> matrix<CPU>::log2(const matrix<CPU> &a)
{
    matrix<CPU> res = create_stacked_matrix(a.rows(), a.columns(), a.height());
    for(size_t i = 0; i < a.elements(); i++)
        res[i] = std::log2(a[i]);
    return res;
}




const float matrix<CPU>::operator[](size_t index) const
{
    return this->data[index];
}

float &matrix<CPU>::operator[](size_t index)
{
    return this->data[index];
}

void matrix<CPU>::set(size_t index, float val)
{
    memcpy(&data[index], &val, sizeof(float));
}

matrix<CPU>& matrix<CPU>::operator=(const matrix<CPU>& other)
{
    if (this != &other) 
    {
        if (owns_memory) 
            memory_pool<CPU>::instance().deallocate(data, n); 
        r = other.r;
        c = other.c;
        n = other.n;
        h = other.h;
        owns_memory = true;
        data = memory_pool<CPU>::instance().allocate(n);
        memcpy(data, other.data, n * sizeof(float));
    }
    return *this;
}

matrix<CPU>& matrix<CPU>::operator=(matrix<CPU>&& other) noexcept
{   
    if (this != &other) 
    {
        if (owns_memory && data) 
            memory_pool<CPU>::instance().deallocate(data, n);
        
        
        r = other.r;
        c = other.c;
        h = other.h;
        n = other.n;
        owns_memory = other.owns_memory;
        data = other.data;

        other.data = nullptr;   
        other.owns_memory = false;
        other.r = other.c = other.n = other.h = 0;
    }
    return *this;
}



matrix<CPU> matrix<CPU>::operator%(const matrix<CPU> &a) const
{
    if(this->r != a.rows() || this->c != a.columns() || (a.height() > 1 && this->h > 1 && a.height() != this->h) )
        throw std::runtime_error("% : matrix shapes need to be equal ");

    if(this->h > 1 && a.height() == 1)
        return bcast_hadamard_to_stacked_matrix(*this, a);

    if(this->h == 1 && a.height() > 1)
        return bcast_hadamard_to_stacked_matrix(a, *this);


    matrix<CPU> res = create_stacked_matrix(a.rows(), a.columns(), a.height());
    for(size_t i = 0; i < a.elements(); i++)
        res[i] = this->data[i] * a[i];
    return res;
    

}

matrix<CPU> matrix<CPU>::operator+(const matrix<CPU> &a) const
{  

    if(this->r != a.rows() || this->c != a.columns() || (a.height() > 1 && this->h > 1 && a.height() != this->h) )
        throw std::runtime_error("+ : matrix shapes need to be equal ");

    if(this->h > 1 && a.height() == 1)
        return bcast_add_to_stacked_matrix(*this, a);

    if(this->h == 1 && a.height() > 1)
        return bcast_add_to_stacked_matrix(a, *this);

    matrix<CPU> res = create_stacked_matrix(a.rows(), a.columns(), a.height());
    for(size_t i = 0; i < a.elements(); i++)
        res[i] = this->data[i] + a[i];
    return res;
}

matrix<CPU> matrix<CPU>::operator+(const float &a) const
{
    matrix<CPU> res = create_stacked_matrix(this->rows(), this->columns(), this->height());
    for(size_t i = 0; i < this->elements(); i++)
        res[i] = this->data[i] + a;
    return res;

}

matrix<CPU> matrix<CPU>::operator-(const matrix<CPU> &a) const
{
    if(this->r != a.rows() || this->c != a.columns() || (a.height() > 1 && this->h > 1 && a.height() != this->h) )
        throw std::runtime_error("- : matrix shapes need to be equal ");

    if(this->h > 1 && a.height() == 1)
        return bcast_add_to_stacked_matrix(*this, a * (-1));

    if(this->h == 1 && a.height() > 1)
        return bcast_add_to_stacked_matrix(a * (-1), *this);

    matrix<CPU> res = create_stacked_matrix(a.rows(), a.columns(), a.height());
    for(size_t i = 0; i < a.elements(); i++)
        res[i] = this->data[i] - a[i];
    return res;

}

matrix<CPU> matrix<CPU>::operator-(const float &a) const
{
    matrix<CPU> res = create_stacked_matrix(this->rows(), this->columns(), this->height());
    for(size_t i = 0; i < this->elements(); i++)
        res[i] = this->data[i] - a;
    return res;
}

matrix<CPU> matrix<CPU>::operator*(const float &a) const
{
    matrix<CPU> res = create_stacked_matrix(this->rows(), this->columns(), this->height());
    for(size_t i = 0; i < this->elements(); i++)
        res[i] = this->data[i] * a;
    return res;

}

matrix<CPU> matrix<CPU>::operator*(const matrix<CPU> &a) const
{    
    if(this->c != a.rows())
        throw std::runtime_error("mat_mul : matrix shapes do not match. ");

    if(this->h > 1 && a.height() == 1)
        return bcast_mat_mul_to_stacked_matrix(*this, a);

    if(this->h == 1 && a.height() > 1)
        return bcast_reversed_mat_mul_to_stacked_matrix(a, *this);

    const size_t rows = this->rows();
    const size_t cols = a.columns();
    const size_t inner = this->columns();


    matrix<CPU> res = create_stacked_matrix(rows, cols, this->h);


    for(size_t layer = 0; layer < this->h; layer++)
        for(size_t row = 0; row < rows; row++) 
        {

            const size_t data_offset = (rows * inner) * layer;
            const size_t a_offset = (inner * cols) * layer;
            const size_t res_offset = (rows * cols) * layer;

            for(size_t col = 0; col < cols; col++)
            {
                float sum = 0;
                for(size_t i = 0; i < inner; i++)
                    sum += this->data[data_offset + row * inner + i] * a[a_offset + i * cols + col];
                res[res_offset + row * cols + col] = sum;
            }
        }

    return res;
}

matrix<CPU> operator*(float val, const matrix<CPU> &a)
{
    return a * val;
}

matrix<CPU> operator+(float val, const matrix<CPU> &a)
{
    return a + val;
}

matrix<CPU> operator-(float val, const matrix<CPU> &a)
{
    return (-1) * a + val;
}



matrix<CPU> matrix<CPU>::operator+=(const matrix<CPU> &a)
{
    *this = *this + a;
    return *this;
}

matrix<CPU> matrix<CPU>::operator-=(const matrix<CPU> &a)
{
    *this = *this - a;
    return *this;
}

matrix<CPU> matrix<CPU>::transpose(const matrix<CPU> &a)
{   


    size_t columns = a.columns();
    size_t rows = a.rows();

    matrix<CPU> res = create_stacked_matrix(columns,rows, a.height());

    for(size_t layer = 0; layer < a.height(); layer++)
    {
        size_t offset = layer * (rows * columns);
        for(size_t row = 0; row < rows; row++)
        {
            for(size_t col = 0; col< columns; col++)
            {
                res[col * res.columns() + row + offset] = a[row * columns + col + offset];
            }
        }
    }
    return res;
}






