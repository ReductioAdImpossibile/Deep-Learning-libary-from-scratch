#include "modelCPU.h"
#include <sstream>




neuralnetwork<CPU>::neuralnetwork() : input_layer_neurons(0)
{}

void neuralnetwork<CPU>::add_layer(const size_t neurons, activation_type atype) 
{

    if(this->imported)
        throw std::runtime_error("You cant modify the structure of imported networks. You can only run, fit, reset weights and check their performance.");

    neurons_per_layer.push_back(neurons);
    afunc_type.push_back(atype);
    this->output_layer_neurons = neurons;
}

void neuralnetwork<CPU>::configure_loss_function(loss_type _ltype) 
{
    if(this->imported)
        throw std::runtime_error("You cant modify the structure of imported networks. You can only run, fit, reset weights and check their performance.");
    this->lfunc_type = _ltype;
}

void neuralnetwork<CPU>::configure_input_layer(const size_t neurons)
{
    if(this->imported)
        throw std::runtime_error("You cant modify the structure of imported networks. You can only run, fit, reset weights and check their performance.");

    this->input_layer_neurons = neurons;
    neurons_per_layer.insert(neurons_per_layer.begin(), neurons);
}





// ------------------------------- BACKPROPAGATION -------------------------------------------

void neuralnetwork<CPU>::initalise_random_weights(float begin, float end)
{
    std::cout << "[INITALISE]" << std::endl;
    if(neurons_per_layer.size() <= 1)
        throw std::runtime_error("Initalise() : run add_layer first to build the NN. ");

    for(int i = 0; i < neurons_per_layer.size()-1; i++)
    {
        size_t rows = neurons_per_layer[i+1];        
        size_t cols = neurons_per_layer[i];      

        std::cout << "[LAYER = " << i << " WEIGHT MATRIX: => ROWS = " << rows << " , COLUMNS = " << cols << ", BIAS = " <<  rows << "] " << std::endl;

        matrix<CPU> mat(rows, cols, begin, end);    
        weight_matrices.push_back(mat);

        matrix<CPU> bias(rows , 1, begin, end);
        bias_matrices.push_back(bias);
    }



    lfunc = loss<CPU>::get_fn(lfunc_type);
    lfunc_dx = loss<CPU>::get_derivative_fn(lfunc_type, afunc_type.back());

    for(size_t a : afunc_type)
    {
        afunc.push_back(activation<CPU>::get_fn(a));
        afunc_dx.push_back(activation<CPU>::get_derivative_fn(a));
    }


    if(afunc_type.back() == activation<CPU>::SOFTMAX && this->lfunc_type != loss<CPU>::CROSS_ENTROPY )
        throw std::invalid_argument("Activation function softmax only works with cross entropy loss function.");

}

void neuralnetwork<CPU>::initalise_xavier_weights()
{
    std::cout << "[INITALISE]" << std::endl;
    if(neurons_per_layer.size() <= 1)
        throw std::runtime_error("Initalise() : run add_layer first to build the NN. ");

    for(int i = 0; i < neurons_per_layer.size()-1; i++)
    {
        size_t rows = neurons_per_layer[i+1];        
        size_t cols = neurons_per_layer[i];      

        std::cout << "[LAYER = " << i << " WEIGHT MATRIX: => ROWS = " << rows << " , COLUMNS = " << cols << ", BIAS = " <<  rows << "] " << std::endl;

        const float range = std::sqrt(6 / (neurons_per_layer[i] + neurons_per_layer[i+1]));
        matrix<CPU> mat(rows, cols, -range, range);    
        weight_matrices.push_back(mat);

        matrix<CPU> bias(rows , 1, -range, range);
        bias_matrices.push_back(bias);
    }



    lfunc = loss<CPU>::get_fn(lfunc_type);
    lfunc_dx = loss<CPU>::get_derivative_fn(lfunc_type, afunc_type.back());

    for(size_t a : afunc_type)
    {
        afunc.push_back(activation<CPU>::get_fn(a));
        afunc_dx.push_back(activation<CPU>::get_derivative_fn(a));
    }


    if(afunc_type.back() == activation<CPU>::SOFTMAX && this->lfunc_type != loss<CPU>::CROSS_ENTROPY )
        throw std::invalid_argument("Activation function softmax only works with cross entropy loss function.");
}

void neuralnetwork<CPU>::initalise_he_weights()
{
        std::cout << "[INITALISE]" << std::endl;
    if(neurons_per_layer.size() <= 1)
        throw std::runtime_error("Initalise() : run add_layer first to build the NN. ");

    for(int i = 0; i < neurons_per_layer.size()-1; i++)
    {
        size_t rows = neurons_per_layer[i+1];        
        size_t cols = neurons_per_layer[i];      

        std::cout << "[LAYER = " << i << " WEIGHT MATRIX: => ROWS = " << rows << " , COLUMNS = " << cols << ", BIAS = " <<  rows << "] " << std::endl;

        const float range = std::sqrt(6 / (neurons_per_layer[i]));
        matrix<CPU> mat(rows, cols, -range, range);    
        weight_matrices.push_back(mat);

        matrix<CPU> bias(rows , 1, -range, range);
        bias_matrices.push_back(bias);
    }



    lfunc = loss<CPU>::get_fn(lfunc_type);
    lfunc_dx = loss<CPU>::get_derivative_fn(lfunc_type, afunc_type.back());

    for(size_t a : afunc_type)
    {
        afunc.push_back(activation<CPU>::get_fn(a));
        afunc_dx.push_back(activation<CPU>::get_derivative_fn(a));
    }


    if(afunc_type.back() == activation<CPU>::SOFTMAX && this->lfunc_type != loss<CPU>::CROSS_ENTROPY )
        throw std::invalid_argument("Activation function softmax only works with cross entropy loss function.");
}

matrix<CPU> neuralnetwork<CPU>::run(const matrix<CPU> &input)
{
    matrix<CPU> result = input;
    
    for(int i = 0; i < neurons_per_layer.size() - 1; i++)
    {

        matrix<CPU> prod = weight_matrices[i] * result + bias_matrices[i];
        
        result = afunc[i](prod);
    }

    return result;
}



std::vector<matrix<CPU>> neuralnetwork<CPU>::layer_outputs(const matrix<CPU> &input)
{
    std::vector<matrix<CPU>> outputs;
    outputs.resize(neurons_per_layer.size());
    outputs[0] = input;

    matrix<CPU> current = outputs[0];
    
    for(int i = 0; i < neurons_per_layer.size() - 1; i++)
    {
        matrix<CPU> prod = weight_matrices[i] * current + bias_matrices[i];
        
        current = afunc[i](prod);

        outputs[i+1] = current;
    }
    return outputs;
}


void neuralnetwork<CPU>::gradient_descent(const size_t steps, dataset<CPU>& ds, double lr , double lambda, size_t batch_size)
{


    std::vector<matrix<CPU>> wgradients;
    wgradients.resize(weight_matrices.size());

    std::vector<matrix<CPU>> bgradients;
    bgradients.resize(bias_matrices.size());

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, ds.input.size() / batch_size - 1); 


    for(size_t i = 0; i < weight_matrices.size(); i++)
    {
        wgradients[i] = matrix<CPU>(weight_matrices[i].rows(), weight_matrices[i].columns(), 0);
        bgradients[i] = matrix<CPU>(bias_matrices[i].rows(), 1, 0);
    }

    for(size_t step = 0; step < steps; step++)
    {

        // ----------------------------------
        for(size_t i = 0; i < weight_matrices.size(); i++)
        {
            wgradients[i].set(0);
            bgradients[i].set(0);
        }

        size_t start = dist(rng);
        std::vector<matrix<CPU>> Z;
        Z.reserve(this->weight_matrices.size() + 1);

        for(size_t pos = start; pos < start + batch_size; pos++)
        {
            matrix<CPU> &input_value = ds.input[pos];
            matrix<CPU> &truth_value = ds.expected[pos];
            Z = layer_outputs(input_value);


            ssize_t index = Z.size() - 1;
            matrix<CPU> delta = lfunc_dx(Z[index], truth_value) % afunc_dx[index-1](weight_matrices[index-1] * Z[index-1] + bias_matrices[index-1]); 

            wgradients[index-1] += delta * Z[index-1].transpose();
            bgradients[index-1] += delta;
            
            index-= 2;
            for(index; index >= 0; index--)  
            {    
                matrix<CPU> weight_transposed = weight_matrices[index+1].transpose();
        
                delta = (weight_transposed * delta) % afunc_dx[index](weight_matrices[index] * Z[index] + bias_matrices[index]);
                
                wgradients[index] += delta * Z[index].transpose();
                bgradients[index] += delta;
            }
            
        }

        const float n = (float) batch_size;
        
        if(lambda != 0.0f)
        {
            for(size_t i = 0; i < weight_matrices.size(); i++)
            {
                weight_matrices[i] = weight_matrices[i] * (1 - lr * lambda / n) - (lr / n) * wgradients[i];
                bias_matrices[i] = bias_matrices[i] * (1 - lr * lambda / n ) - (lr / n) * bgradients[i];
            }
        }
        else
        {
            for(size_t i = 0; i < weight_matrices.size(); i++)
            {
                weight_matrices[i] -=  (lr / n)  * wgradients[i];
                bias_matrices[i] -=  (lr / n) * bgradients[i];
            }
        }


    }
}






// ------------------------------------------------------------------------------------

void neuralnetwork<CPU>::fit(const size_t epochs, dataset<CPU>& ds, optimizer_type ofunc, double lr, size_t batch_size )
{

    size_t steps;

    switch(ofunc)
    {
        case optimizer<CPU>::STOCHASTIC_GRADIENT_DESCENT:
            steps = epochs * ds.input.size(); 
            gradient_descent(steps, ds, lr, 0.0f, 1);
            break;

        case optimizer<CPU>::BATCH_GRADIENT_DESCENT:
            steps = epochs;
            gradient_descent(steps, ds, lr, 0.0f, ds.input.size());
            break;

        case optimizer<CPU>::MIN_BATCH_GRADIENT_DESCENT:
            steps = epochs * (ds.input.size() / batch_size);
            gradient_descent(steps, ds, lr, 0.0f, batch_size);
            break;
    }
}


void neuralnetwork<CPU>::fit(const size_t epochs, dataset<CPU> &ds, optimizer_type ofunc, hyperparameter<CPU>& param)
{

    size_t steps;
    switch(ofunc)
    {
        case optimizer<CPU>::STOCHASTIC_GRADIENT_DESCENT:
            steps = epochs * ds.input.size(); 
            gradient_descent(steps, ds, param.lr, param.lambda, 1);
            break;

        case optimizer<CPU>::BATCH_GRADIENT_DESCENT:
            steps = epochs;
            gradient_descent(steps, ds, param.lr, param.lambda, ds.input.size());
            break;

        case optimizer<CPU>::MIN_BATCH_GRADIENT_DESCENT:
            steps = epochs * (ds.input.size() / param.batch_size);
            gradient_descent(steps, ds, param.lr, param.lambda, param.batch_size);
            break;
    }
}


void neuralnetwork<CPU>::fit(const size_t epochs, dataset<CPU> &ds, adam_optimizer<CPU> &adam)
{

    const size_t steps = epochs * (ds.input.size() / adam.batch_size);

    std::vector<matrix<CPU>> weight_gradients;
    weight_gradients.resize(weight_matrices.size());

    std::vector<matrix<CPU>> bias_gradients;
    bias_gradients.resize(bias_matrices.size());


    std::vector<matrix<CPU>> weight_momentum;
    weight_momentum.resize(weight_matrices.size());

    std::vector<matrix<CPU>> bias_momentum;
    bias_momentum.resize(weight_matrices.size());

    std::vector<matrix<CPU>> weight_variance;
    weight_variance.resize(weight_matrices.size());
        
    std::vector<matrix<CPU>> bias_variance;
    bias_variance.resize(weight_matrices.size());


    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, ds.input.size() / adam.batch_size - 1); 


    for(size_t i = 0; i < weight_matrices.size(); i++)
    {
        weight_gradients[i] = matrix<CPU>(weight_matrices[i].rows(), weight_matrices[i].columns(), 0);
        bias_gradients[i] = matrix<CPU>(bias_matrices[i].rows(), 1, 0);

        weight_momentum[i] = matrix<CPU>(weight_matrices[i].rows(), weight_matrices[i].columns(), 0);
        weight_variance[i] = matrix<CPU>(weight_matrices[i].rows(), weight_matrices[i].columns(), 0);

        bias_momentum[i] = matrix<CPU>(bias_matrices[i].rows(), 1, 0);
        bias_variance[i] = matrix<CPU>(bias_matrices[i].rows(), 1, 0);
    }

    for(size_t step = 1; step <= steps; step++)
    {

        for(size_t i = 0; i < weight_matrices.size(); i++)
        {
            weight_gradients[i].set(0);
            bias_gradients[i].set(0);
        }

        size_t start = dist(rng);
        std::vector<matrix<CPU>> Z;
        Z.reserve(this->weight_matrices.size() + 1);

        for(size_t pos = start; pos < start + adam.batch_size; pos++)
        {
            matrix<CPU> &input_value = ds.input[pos];
            matrix<CPU> &truth_value = ds.expected[pos];
            Z = layer_outputs(input_value);


            ssize_t index = Z.size() - 1;
            matrix<CPU> delta = lfunc_dx(Z[index], truth_value) % afunc_dx[index-1](weight_matrices[index-1] * Z[index-1] + bias_matrices[index-1]); 

            weight_gradients[index-1] += delta * Z[index-1].transpose();
            bias_gradients[index-1] += delta;
            
            index-= 2;
            for(index; index >= 0; index--)  
            {    
                matrix<CPU> weight_transposed = weight_matrices[index+1].transpose();
        
                delta = (weight_transposed * delta) % afunc_dx[index](weight_matrices[index] * Z[index] + bias_matrices[index]);
                
                weight_gradients[index] += delta * Z[index].transpose();
                bias_gradients[index] += delta;
            }
            
        }

        const float n = (float) adam.batch_size;
        for(size_t i = 0; i < weight_matrices.size(); i++)
        {

            weight_gradients[i] = weight_gradients[i] * (1 / n);
            bias_gradients[i] = bias_gradients[i] * (1 / n);
            
            weight_momentum[i] = adam.beta1 * weight_momentum[i] + (1 - adam.beta1) * weight_gradients[i];
            weight_variance[i] = adam.beta2 * weight_variance[i] + (1 - adam.beta2) * matrix<CPU>::square(weight_gradients[i]);
            
            matrix<CPU> weight_momentum_comp = weight_momentum[i] * (1 / (1-std::pow(adam.beta1, step))); 
            matrix<CPU> weight_variance_comp = weight_variance[i] * (1 / (1-std::pow(adam.beta2, step))); 

            weight_matrices[i] -= adam.lr * (weight_momentum_comp  % matrix<CPU>::reciprocal(matrix<CPU>::sqrt(weight_variance_comp) + adam.epsilon)); 


            bias_momentum[i] = adam.beta1 * bias_momentum[i] + (1 - adam.beta1) * bias_gradients[i];
            bias_variance[i] = adam.beta2 * bias_variance[i] + (1 - adam.beta2) * matrix<CPU>::square(bias_gradients[i]);

            matrix<CPU> bias_momentum_comp = bias_momentum[i] * (1 / (1-std::pow(adam.beta1, step))); 
            matrix<CPU> bias_variance_comp = bias_variance[i] * (1 / (1-std::pow(adam.beta2, step))); 

            bias_matrices[i] -= adam.lr * bias_momentum_comp  % matrix<CPU>::reciprocal(matrix<CPU>::sqrt(bias_variance_comp) + adam.epsilon);


            // AdamW enabled : 
            if(adam.lambda != 0)
            {
                weight_matrices[i] -= weight_matrices[i] *  adam.lr * adam.lambda;
                bias_matrices[i] -= bias_matrices[i] * adam.lr * adam.lambda;
            }
        }

    }
}

void neuralnetwork<CPU>::performance(dataset<CPU> &ds, std::string name)
{

    double rmsqe = 0;
    double accuracy = 0;

    
    #pragma omp parallel for reduction(+ : rmsqe, accuracy) num_threads(4)
    for(size_t i = 0; i < ds.input.size(); i++)
    {
        matrix<CPU> pred = run(ds.input[i]);
        matrix<CPU> diff = (pred - ds.expected[i]);
        rmsqe += diff.L2();
        accuracy += (pred.argmax() == ds.expected[i].argmax());
    }   

    rmsqe = std::sqrt(rmsqe / ds.input.size());
    accuracy /= ds.input.size();

    std::cout << "[PERFORMANCE RESULTS FOR DATASET : " << name << "]" << std::endl;
    std::cout << "[ => Accuracy : " << accuracy << "]" <<  std::endl;
    std::cout << "[ => RMSQE : " << rmsqe << "]" << std::endl;

}

void neuralnetwork<CPU>::performance(dataset<CPU> &ds)
{
    performance(ds, "");
}

void neuralnetwork<CPU>::load_weights(const std::string &filename)
{
    std::string line;
    std::fstream file(filename);


    if(!file.is_open())
        throw std::runtime_error("load_weights : File could not be openend!");


    size_t neurons_per_layer_size;
    file >> neurons_per_layer_size;

    size_t afunc_type_size;
    file >> afunc_type_size;

    size_t weights_size;
    file >> weights_size;

    
    this->neurons_per_layer.resize(neurons_per_layer_size);
    this->afunc_type.resize(afunc_type_size);
    
    this->weight_matrices.resize(weights_size);
    this->bias_matrices.resize(weights_size);


    for(size_t i = 0; i < this->neurons_per_layer.size(); i++)
        if(!(file >> neurons_per_layer[i]))
            throw std::runtime_error("load_weights : Error when reading neurons per layer " );


    for(size_t i = 0; i < this->afunc_type.size(); i++)
        if(!(file >> afunc_type[i]))
            throw std::runtime_error("load_weights : Error when reading activation functions " );

    if(!(file >> lfunc_type))
        throw std::runtime_error("load_weights : Error when reading loss function " );



    for(size_t m = 0; m < this->weight_matrices.size(); m++)
    {
        size_t rows, cols;
        if(!(file >> rows >> cols))
            throw std::runtime_error("load_weights : Error when reading weight dim's");
        
        std::vector<float> values;
        values.resize(rows * cols);
        for(size_t i = 0; i < rows * cols; i++)
            if(!(file >> values[i]))
                throw std::runtime_error("load_weights : Error when reading weight values"); 

        this->weight_matrices[m] = matrix<CPU>(rows, cols, values);  
    }

    for(size_t m = 0; m < this->bias_matrices.size(); m++)
    {
        size_t rows, cols;
        if(!(file >> rows >> cols))
            throw std::runtime_error("load_weights : Error when reading bias dim's");
        
        std::vector<float> values;
        values.resize(rows * cols);
        for(size_t i = 0; i < rows * cols; i++)
            if(!(file >> values[i]))
                throw std::runtime_error("load_weights : Error when reading bias values"); 

        this->bias_matrices[m] = matrix<CPU>(rows, cols, values);  
    }


    this->lfunc = loss<CPU>::get_fn(this->lfunc_type);
    this->lfunc_dx = loss<CPU>::get_derivative_fn(this->lfunc_type, afunc_type.back());

    for(size_t a : afunc_type)
    {
        this->afunc.push_back(activation<CPU>::get_fn(a));
        this->afunc_dx.push_back(activation<CPU>::get_derivative_fn(a));
    }

    this->imported = true;
    std::cout << "[IMPORTED " << filename << " SUCCESSFULLY ]" << std::endl;

}

void neuralnetwork<CPU>::save_weights(const std::string &filename)
{
    std::ofstream file(filename);

    if(!file.is_open())
        throw std::runtime_error("save_weights : File could not be created!");


    file << neurons_per_layer.size() << "\n";
    file << afunc_type.size() << "\n";
    file << weight_matrices.size() << "\n";

    for(size_t neurons : this->neurons_per_layer)
        file << neurons << " ";


    for(size_t activation_function : this->afunc_type)
        file << activation_function << " ";
    file << lfunc_type << " " << "\n";


    for(matrix<CPU> &mat : this->weight_matrices )
    {
        file << mat.rows() << " " << mat.columns() << "\n";
        for(float value : mat )
            file << value << " ";
        file << "\n";
    }

    for(matrix<CPU> &bias : this->bias_matrices )
    {
        file << bias.rows() << " " << bias.columns() << "\n";
        for(float value : bias )
            file << value << " ";
        file << "\n";
    }

}
