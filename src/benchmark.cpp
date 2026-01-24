#include "benchmark.h"
#include <chrono>

volatile double sink;
using Clock = std::chrono::steady_clock;
using time_point = Clock::time_point;
using ms = std::chrono::duration<double, std::milli>;

double get_time(time_point start, time_point end)
{
    return std::chrono::duration_cast<ms>(end - start).count();
}

double relative_error(double res_tensor, double res_seq)
{
    return std::sqrt((res_tensor - res_seq) * (res_tensor - res_seq) / (res_seq * res_seq));
}

double relative_error(Tensor& res_tensor, std::vector<float>& res_seq)
{
    double quadratic_sum = 0;
    double quadratic_difference = 0;
    for(int i = 0; i < res_seq.size(); i++)
    {
        double v = res_tensor[i] - res_seq[i];
        quadratic_difference += v * v;
        quadratic_sum += res_seq[i] * res_seq[i]; 
    }
    return std::sqrt(quadratic_difference / quadratic_sum);
}


double benchmark_sum(const std::vector<float>& v) 
{
    double s = 0.0;
    for (double x : v)
        s += x;
    return s;
}

double benchmark_l1(const std::vector<float>& v) 
{
    double s = 0.0;
    for (double x : v)
        s += std::abs(x);
    return s;
}

double benchmark_l2(const std::vector<float>& v) 
{
    double s = 0.0;
    for (double x : v)
        s += x * x;
    return std::sqrt(s);
}

double benchmark_product(const std::vector<float>& v)
{
    double p = 1.0;
    for (double x : v)
        p *= x;
    return p;
}

void benchmark_add(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) 
{
    for (std::size_t i = 0; i < a.size(); ++i)
        out[i] = a[i] + b[i];
    sink = out[0];
}

void benchmark_sub(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) {
    for (std::size_t i = 0; i < a.size(); ++i)
        out[i] = a[i] - b[i];
    sink = out[0];
}

void benchmark_hadamard(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) {
    for (std::size_t i = 0; i < a.size(); ++i)
        out[i] = a[i] * b[i];
    sink = out[0];
}

void benchmark_scale(const std::vector<float>& v, std::vector<float>& out, double alpha) {
    for (std::size_t i = 0; i < v.size(); ++i)
        out[i] = alpha * v[i];
    sink = out[0];
}

void benchmark_tensor_run()
{
    const size_t N = 200000000;
    const size_t iters = 10;

    Tensor tensor1({N,1}, -10, 10);
    Tensor tensor2({N,1}, -10, 10);
    Tensor result({N,1});

    std::vector<float> tensor1_v = tensor1.vector();
    std::vector<float> tensor2_v = tensor2.vector();
    std::vector<float> result_v(N);

    time_point start, end;
    float res_seq, res_tensor;

    std::vector<double> times_seq(7);
    std::vector<double> times_tensor(7);
    std::vector<double> relative_erros(7);
    std::vector<std::string> labels = {"tensor sum         ", "L1 of tensor       ", "L2 of tensor       ", "hadamard product   ", "tensor addition    ", "tensor substraction", "tensor scaling     "  };


    // -------------------- SUM --------------------

    std::cout << "Running Sum ... " << std::endl;
    start = Clock::now();
    res_seq = benchmark_sum(tensor1_v);
    end = Clock::now();
    times_seq[0] = get_time(start, end);


    start = Clock::now();
    res_tensor = tensor1.sum();
    end = Clock::now();
    times_tensor[0] = get_time(start, end);

    relative_erros[0] = relative_error(res_tensor, res_seq);


    
    // -------------------- L1 -----------------------
    std::cout << "Running L1 ... " << std::endl;
    start = Clock::now();
    res_seq = benchmark_l1(tensor1_v);
    end = Clock::now();
    times_seq[1] = get_time(start, end);

    start = Clock::now();
    res_tensor = tensor1.L1();
    end = Clock::now();
    times_tensor[1] = get_time(start, end);

    relative_erros[1] = relative_error(res_tensor, res_seq);

    
    
    // ------------------l2 ----------------------
    std::cout << "Running L2 ... " << std::endl;
    start = Clock::now();
    res_seq = benchmark_l2(tensor1_v);
    end = Clock::now();
    times_seq[2] = get_time(start, end);

    start = Clock::now();
    res_tensor = tensor1.L2();
    end = Clock::now();
    times_tensor[2] = get_time(start, end);

    relative_erros[2] = relative_error(res_tensor, res_seq);

    
    // --------------------- hadamard product --------------------
    std::cout << "Running hadamard product ... " << std::endl;
    start = Clock::now();
    benchmark_hadamard(tensor1_v, tensor2_v, result_v);
    end = Clock::now();
    times_seq[3] = get_time(start, end);

    start = Clock::now();
    Tensor::hadamard(tensor1, tensor2, result);
    end = Clock::now();
    times_tensor[3] = get_time(start, end);

    relative_erros[3] = relative_error(result, result_v);


    
    // ------------------------- add -------------------------
    std::cout << "Running add ... " << std::endl;
    start = Clock::now();
    benchmark_add(tensor1_v, tensor2_v, result_v);
    end = Clock::now();
    times_seq[4] = get_time(start, end);

    start = Clock::now();
    Tensor::add(tensor1, tensor2, result);
    end = Clock::now();
    times_tensor[4] = get_time(start, end);

    relative_erros[4] = relative_error(result, result_v);




    // --------------------- sub -------------------------
    std::cout << "Running sub ... " << std::endl;
    start = Clock::now();
    benchmark_sub(tensor1_v, tensor2_v, result_v);
    end = Clock::now();
    times_seq[5] = get_time(start, end);

    start = Clock::now();
    Tensor::sub(tensor1, tensor2, result);
    end = Clock::now();
    times_tensor[5] = get_time(start, end);

    relative_erros[5] = relative_error(result, result_v);





    // ------------------ scale ------------------
    std::cout << "Running scale ... " << std::endl;
    start = Clock::now();
    benchmark_scale(tensor1_v, result_v, 0.61);
    end = Clock::now();
    times_seq[6] = get_time(start, end);

    start = Clock::now();
    Tensor::scale(tensor1, 0.61, result);
    end = Clock::now();
    times_tensor[6] = get_time(start, end);

    relative_erros[6] = relative_error(result, result_v);

    std::cout << " ================== RESULTS ================ " << std::endl;

    std::cout << "Operation     " << "       Speed up    " << "Relative Error    " << std::endl;
    for(int i = 0; i < relative_erros.size(); i++)
    {
        std::cout << labels.at(i) << " " << times_seq[i] / times_tensor[i] << "      " << relative_erros[i] << std::endl; 
    }
   
}