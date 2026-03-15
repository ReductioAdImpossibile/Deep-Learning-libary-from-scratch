#include "DeepModel.h"
#include <sstream>
#include <iomanip>


NeuralNetwork::NeuralNetwork() : input_layer_neurons(0), output_layer_neurons(0)
{
    this->loss_function_class = Loss();
}

void NeuralNetwork::add_layer(const size_t neurons, activation_type atype) 
{

    if(this->imported)
        throw std::runtime_error("You cant modify the structure of imported networks. You can only run, fit, reset weights and check their performance.");

    neurons_per_layer.push_back(neurons);
    afunc_type.push_back(atype);
    this->output_layer_neurons = neurons;
}

void NeuralNetwork::configure_loss_function(loss_type _ltype) 
{
    if(this->imported)
        throw std::runtime_error("You cant modify the structure of imported networks. You can only run, fit, reset weights and check their performance.");
    this->lfunc_type = _ltype;
}

void NeuralNetwork::set_loss_weights(const std::vector<float> w)
{
    if(output_layer_neurons == 0)
        throw std::runtime_error("set_loss_weights : Fully configure the Network first bevore setting the loss weights.");
    if(output_layer_neurons != w.size())
        throw std::runtime_error("set_loss_weight : Weight size needs to be equal to output layer size");
    
    this->loss_function_class.weights = Matrix::create_stacked_matrix(w.size(), 1, 1 , w);
}

void NeuralNetwork::configure_input_layer(const size_t neurons)
{
    if(this->imported)
        throw std::runtime_error("You cant modify the structure of imported networks. You can only run, fit, reset weights and check their performance.");

    this->input_layer_neurons = neurons;
    neurons_per_layer.insert(neurons_per_layer.begin(), neurons);
}





// ------------------------------- BACKPROPAGATION -------------------------------------------

void NeuralNetwork::initalise_random_weights(float begin, float end)
{
    std::cout << "[INITALISE]" << std::endl;
    if(neurons_per_layer.size() <= 1)
        throw std::runtime_error("Initalise() : run add_layer first to build the NN. ");

    for(int i = 0; i < neurons_per_layer.size()-1; i++)
    {
        size_t rows = neurons_per_layer[i+1];        
        size_t cols = neurons_per_layer[i];      

        std::cout << "[LAYER = " << i << " WEIGHT Matrix: => ROWS = " << rows << " , COLUMNS = " << cols << ", BIAS = " <<  rows << "] " << std::endl;

        Matrix mat(rows, cols, begin, end);    
        weight_matrices.push_back(mat);

        Matrix bias(rows , 1, begin, end);
        bias_matrices.push_back(bias);
    }



    if(this->loss_function_class.weights.empty())
        this->loss_function_class.weights = Matrix::create_stacked_matrix(this->output_layer_neurons,1, 1, 1);

    lfunc = loss_function_class.get_fn(lfunc_type);
    lfunc_dx = loss_function_class.get_derivative_fn(lfunc_type, afunc_type.back());

    for(size_t a : afunc_type)
    {
        afunc.push_back(Activation::get_fn(a));
        afunc_dx.push_back(Activation::get_derivative_fn(a));
    }


    if(afunc_type.back() == Activation::SOFTMAX && this->lfunc_type != Loss::CROSS_ENTROPY )
        throw std::invalid_argument("Activation function softmax only works with cross entropy loss function.");

}

void NeuralNetwork::initalise_xavier_weights()
{
    std::cout << "[INITALISE]" << std::endl;
    if(neurons_per_layer.size() <= 1)
        throw std::runtime_error("Initalise() : run add_layer first to build the NN. ");

    for(int i = 0; i < neurons_per_layer.size()-1; i++)
    {
        size_t rows = neurons_per_layer[i+1];        
        size_t cols = neurons_per_layer[i];      

        std::cout << "[LAYER = " << i << " WEIGHT Matrix: => ROWS = " << rows << " , COLUMNS = " << cols << ", BIAS = " <<  rows << "] " << std::endl;

        const float range = std::sqrt(6 / (neurons_per_layer[i] + neurons_per_layer[i+1]));
        Matrix mat(rows, cols, -range, range);    
        weight_matrices.push_back(mat);

        Matrix bias(rows , 1, -range, range);
        bias_matrices.push_back(bias);
    }


    if(this->loss_function_class.weights.empty())
        this->loss_function_class.weights = Matrix(this->output_layer_neurons,1, 1,  1);

    lfunc = loss_function_class.get_fn(lfunc_type);
    lfunc_dx = loss_function_class.get_derivative_fn(lfunc_type, afunc_type.back());

    for(size_t a : afunc_type)
    {
        afunc.push_back(Activation::get_fn(a));
        afunc_dx.push_back(Activation::get_derivative_fn(a));
    }


    if(afunc_type.back() == Activation::SOFTMAX && this->lfunc_type != Loss::CROSS_ENTROPY )
        throw std::invalid_argument("Activation function softmax only works with cross entropy loss function.");
}

void NeuralNetwork::initalise_he_weights()
{
    std::cout << "[INITALISE]" << std::endl;
    if(neurons_per_layer.size() <= 1)
        throw std::runtime_error("Initalise() : run add_layer first to build the NN. ");

    for(int i = 0; i < neurons_per_layer.size()-1; i++)
    {
        size_t rows = neurons_per_layer[i+1];        
        size_t cols = neurons_per_layer[i];      

        std::cout << "[LAYER = " << i << " WEIGHT Matrix: => ROWS = " << rows << " , COLUMNS = " << cols << ", BIAS = " <<  rows << "] " << std::endl;

        const float range = std::sqrt(6 / (neurons_per_layer[i]));
        Matrix mat(rows, cols, -range, range);    
        weight_matrices.push_back(mat);

        Matrix bias(rows , 1, -range, range);
        bias_matrices.push_back(bias);
    }



    if(this->loss_function_class.weights.empty())
        this->loss_function_class.weights = Matrix(this->output_layer_neurons, 1, 1,  1);

    lfunc = loss_function_class.get_fn(lfunc_type);
    lfunc_dx = loss_function_class.get_derivative_fn(lfunc_type, afunc_type.back());

    for(size_t a : afunc_type)
    {
        afunc.push_back(Activation::get_fn(a));
        afunc_dx.push_back(Activation::get_derivative_fn(a));
    }


    if(afunc_type.back() == Activation::SOFTMAX && this->lfunc_type != Loss::CROSS_ENTROPY )
        throw std::invalid_argument("Activation function softmax only works with cross entropy loss function.");
}

Matrix NeuralNetwork::run(const Matrix &input)
{
    Matrix result = input;
    
    for(int i = 0; i < neurons_per_layer.size() - 1; i++)
    {
        Matrix prod = weight_matrices[i] * result + bias_matrices[i];
        result = afunc[i](prod);   
    }

    return result;
}

std::vector<Matrix> NeuralNetwork::layer_outputs(const Matrix &input)
{
    std::vector<Matrix> outputs;
    outputs.resize(neurons_per_layer.size());
    outputs[0] = input;

    Matrix current = outputs[0];
    
    for(int i = 0; i < neurons_per_layer.size() - 1; i++)
    {
        Matrix prod = weight_matrices[i] * current;
        prod += bias_matrices[i];
        current = afunc[i](prod);
        outputs[i+1] = current;
    }
    return outputs;
}


void NeuralNetwork::gradient_descent(const size_t steps, Dataset& ds, double lr , double lambda, size_t batch_size)
{


    std::cout << "==================[TRAINING]=====================" << std::endl;

    size_t current_epoch = 0;
    std::vector<Matrix> wgradients;
    wgradients.resize(weight_matrices.size());

    std::vector<Matrix> bgradients;
    bgradients.resize(bias_matrices.size());

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, ds.input.height() / batch_size - 1); 

    for(size_t i = 0; i < weight_matrices.size(); i++)
    {
        wgradients[i] = Matrix::create_stacked_matrix(weight_matrices[i].rows(), weight_matrices[i].columns(), batch_size, 0.0f);
        bgradients[i] = Matrix::create_stacked_matrix(bias_matrices[i].rows(), 1, batch_size, 0.0f);
    }

    for(size_t step = 1; step <= steps; step++)
    {

        print_status(step, steps, batch_size, ds.input.height(), current_epoch);

        for(size_t i = 0; i < weight_matrices.size(); i++)
        {
            wgradients[i].set(0);
            bgradients[i].set(0);
        }

        size_t batchidx = dist(rng);
        size_t start = batchidx * batch_size;
        size_t end   = std::min(start + batch_size, ds.input.height());

        std::vector<Matrix> Z;
 
        Matrix input = ds.input.slice_stacked_matrix(start, end); 
        Matrix truth = ds.expected.slice_stacked_matrix(start, end); 

        Z = layer_outputs(input);
        ssize_t index = Z.size() - 1;

        Matrix delta = lfunc_dx(Z[index], truth) % afunc_dx[index-1](weight_matrices[index-1] * Z[index-1] + bias_matrices[index-1]);
        

        wgradients[index-1] = delta * Matrix::transpose(Z[index-1]);
        bgradients[index-1] = delta;

        index-= 2;
        for(; index >= 0; index--)                  
        {    
            Matrix weight_transposed = Matrix::transpose(weight_matrices[index+1]);
        
            delta = (weight_transposed * delta) % afunc_dx[index](weight_matrices[index] * Z[index] + bias_matrices[index]);
                
            wgradients[index] = delta * Matrix::transpose(Z[index]);
            bgradients[index] = delta;
        }
         

        const float n = (float) batch_size;
        if(lambda != 0.0f)
        {
            for(size_t i = 0; i < weight_matrices.size(); i++)
            {
                weight_matrices[i] -= weight_matrices[i] * (1 - lr * lambda / n) - lr * (1/n * Matrix::reduce_sum(wgradients[i]));
                bias_matrices[i] -= bias_matrices[i] * (1 - lr * lambda / n ) - lr * (1/n * Matrix::reduce_sum(bgradients[i]));
            }
        }
        else
        {
            for(size_t i = 0; i < weight_matrices.size(); i++)
            {
                weight_matrices[i] -=  (lr / n)  * Matrix::reduce_sum(wgradients[i]);
                bias_matrices[i] -=  (lr / n) * Matrix::reduce_sum(bgradients[i]);
            }
        }
    }


    std::cout << std::endl;

    std::cout << "=================================================" << std::endl;
}



// ------------------------------------------------------------------------------------




void NeuralNetwork::fit(const size_t epochs, Dataset &ds, ADAM_Optimizer &adam)
{

    std::cout << "==================[TRAINING]=====================" << std::endl;
    size_t steps = epochs * (ds.input.height() / adam.batch_size);
    size_t current_epoch = 0;

    std::vector<Matrix> weight_gradients;
    weight_gradients.resize(weight_matrices.size());

    std::vector<Matrix> bias_gradients;
    bias_gradients.resize(bias_matrices.size());


    std::vector<Matrix> weight_momentum;
    weight_momentum.resize(weight_matrices.size());

    std::vector<Matrix> bias_momentum;
    bias_momentum.resize(weight_matrices.size());

    std::vector<Matrix> weight_variance;
    weight_variance.resize(weight_matrices.size());
        
    std::vector<Matrix> bias_variance;
    bias_variance.resize(weight_matrices.size());


    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, ds.input.height() / adam.batch_size - 1); 


    for(size_t i = 0; i < weight_matrices.size(); i++)
    {
        weight_gradients[i] = Matrix::create_stacked_matrix(weight_matrices[i].rows(), weight_matrices[i].columns(), adam.batch_size, 0.0f);
        bias_gradients[i] = Matrix::create_stacked_matrix(bias_matrices[i].rows(), 1, adam.batch_size, 0.0f);

        weight_momentum[i] = Matrix(weight_matrices[i].rows(), weight_matrices[i].columns(), 0);
        weight_variance[i] = Matrix(weight_matrices[i].rows(), weight_matrices[i].columns(), 0);

        bias_momentum[i] = Matrix(bias_matrices[i].rows(), 1, 0);
        bias_variance[i] = Matrix(bias_matrices[i].rows(), 1, 0);
    }


    for(size_t step = 1; step <= steps; step++)
    {   
        
        print_status(step, steps, adam.batch_size, ds.input.height(), current_epoch);
        
        for(size_t i = 0; i < weight_matrices.size(); i++)
        {
            weight_gradients[i].set(0);
            bias_gradients[i].set(0);
        }



        size_t batchidx = dist(rng);
        size_t start = batchidx * adam.batch_size;
        size_t end   = std::min(start + adam.batch_size, ds.input.height());

 
        Matrix input = ds.input.slice_stacked_matrix(start, end); 
        Matrix truth = ds.expected.slice_stacked_matrix(start, end); 

        std::vector<Matrix> Z;
        Z.reserve(this->weight_matrices.size() + 1);
        Z = layer_outputs(input);
        
        ssize_t index = Z.size() - 1;
        

        Matrix delta = lfunc_dx(Z[index], truth) % afunc_dx[index-1](weight_matrices[index-1] * Z[index-1] + bias_matrices[index-1]);

        weight_gradients[index-1] = delta * Matrix::transpose(Z[index-1]);
        bias_gradients[index-1] = delta;


        index-= 2;
        for(; index >= 0; index--)                  
        {    
            Matrix weight_transposed = Matrix::transpose(weight_matrices[index+1]);
        
            delta = (weight_transposed * delta) % afunc_dx[index](weight_matrices[index] * Z[index] + bias_matrices[index]);
                
            weight_gradients[index] = delta * Matrix::transpose(Z[index]);
            bias_gradients[index] = delta;
        }


        const float n = (float) adam.batch_size;
        for(size_t i = 0; i < weight_matrices.size(); i++)
        {

            weight_gradients[i] = weight_gradients[i] * (1 / n);
            bias_gradients[i] = bias_gradients[i] * (1 / n);

            

            Matrix weight_gradients_reduced = Matrix::reduce_sum(weight_gradients[i]);

            weight_momentum[i] = adam.beta1 * weight_momentum[i] + (1 - adam.beta1) * weight_gradients_reduced;
            weight_variance[i] = adam.beta2 * weight_variance[i] + (1 - adam.beta2) * Matrix::square(weight_gradients_reduced);
            
            Matrix weight_momentum_comp = weight_momentum[i] * (1 / (1-std::pow(adam.beta1, step))); 
            Matrix weight_variance_comp = weight_variance[i] * (1 / (1-std::pow(adam.beta2, step))); 

            weight_matrices[i] -= adam.lr * (weight_momentum_comp  % Matrix::reciprocal(Matrix::sqrt(weight_variance_comp) + adam.epsilon)); 




            Matrix bias_gradients_reduced = Matrix::reduce_sum(bias_gradients[i]);

            bias_momentum[i] = adam.beta1 * bias_momentum[i] + (1 - adam.beta1) * bias_gradients_reduced;
            bias_variance[i] = adam.beta2 * bias_variance[i] + (1 - adam.beta2) * Matrix::square(bias_gradients_reduced);

            Matrix bias_momentum_comp = bias_momentum[i] * (1 / (1-std::pow(adam.beta1, step))); 
            Matrix bias_variance_comp = bias_variance[i] * (1 / (1-std::pow(adam.beta2, step))); 

            bias_matrices[i] -= adam.lr * bias_momentum_comp  % Matrix::reciprocal(Matrix::sqrt(bias_variance_comp) + adam.epsilon);


            // AdamW enabled :
            if(adam.lambda != 0)
            {
                weight_matrices[i] -= weight_matrices[i] *  adam.lr * adam.lambda;
                bias_matrices[i] -= bias_matrices[i] * adam.lr * adam.lambda;
            }
        }

    }

    std::cout << std::endl;
    std::cout << "=================================================" << std::endl;

}


void NeuralNetwork::print_status(const size_t step, const size_t steps, const size_t batch_size, const size_t dataset_size, size_t& current_epoch)
{
    size_t batch_index = step % (dataset_size / batch_size);
    if(batch_index == 0)
        current_epoch++;

    if(step % 1000 == 0 || batch_index == 0)
    { 
        float percentage = static_cast<float>(step) / static_cast<float>(steps);
        std::cout << "\r[Epoch : " << current_epoch << " , " << std::fixed << std::setprecision(2) << percentage * 100 << " % finished ]"  << std::flush;
    }
}



void NeuralNetwork::performance(Dataset &ds, std::string name)
{
    
    float accuracy = 0; 

    std::vector<size_t> p = run(ds.input).argmax();
    std::vector<size_t> e = ds.expected.argmax();

    for(int i = 0; i < ds.input.height(); i++)
    {

        accuracy += (p[i] == e[i]);
    }
    

    accuracy /= ds.input.height();

    std::cout << "[PERFORMANCE RESULTS FOR Dataset : " << name << "]" << std::endl;
    std::cout << "[ => Accuracy : " << accuracy << "]" <<  std::endl;

}


void NeuralNetwork::fit(const size_t epochs, Dataset& ds, optimizer_type ofunc, double lr, size_t batch_size )
{

    size_t steps;

    switch(ofunc)
    {
        case Optimizer::STOCHASTIC_GRADIENT_DESCENT:
            steps = epochs * ds.input.height(); 
            gradient_descent(steps, ds, lr, 0.0f, 1);
            break;

        case Optimizer::BATCH_GRADIENT_DESCENT:
            steps = epochs;
            gradient_descent(steps, ds, lr, 0.0f, ds.input.height());
            break;

        case Optimizer::MIN_BATCH_GRADIENT_DESCENT:
            steps = epochs * (ds.input.height() / batch_size);
            gradient_descent(steps, ds, lr, 0.0f, batch_size);
            break;
    }
}


void NeuralNetwork::fit(const size_t epochs, Dataset &ds, optimizer_type ofunc, Hyperparameter& param)
{

    size_t steps;
    switch(ofunc)
    {
        case Optimizer::STOCHASTIC_GRADIENT_DESCENT:
            steps = epochs * ds.input.height(); 
            gradient_descent(steps, ds, param.lr, param.lambda, 1);
            break;

        case Optimizer::BATCH_GRADIENT_DESCENT:
            steps = epochs;
            gradient_descent(steps, ds, param.lr, param.lambda, ds.input.height());
            break;

        case Optimizer::MIN_BATCH_GRADIENT_DESCENT:
            steps = epochs * (ds.input.height() / param.batch_size);
            gradient_descent(steps, ds, param.lr, param.lambda, param.batch_size);
            break;
    }
}

void NeuralNetwork::performance(Dataset &ds)
{
    performance(ds, "");
}

void NeuralNetwork::binary_confusion_matrix(Dataset &ds, const float threshold)
{
/*
    if(output_layer_neurons != 2)
        throw std::runtime_error("binary_confusion_Matrix : output needs to be binary.");

    int TP = 0, FP = 0, TN = 0, FN = 0;

    for(size_t i = 0; i < ds.input.size(); i++)
    {
        Matrix pred = run(ds.input[i]);

        int pred_class   = pred[1] >= threshold ? 1 : 0;
        int actual_class = ds.expected[i].argmax()[0];

        if      (actual_class == 1 && pred_class == 1) TP++;
        else if (actual_class == 0 && pred_class == 1) FP++;
        else if (actual_class == 0 && pred_class == 0) TN++;
        else if (actual_class == 1 && pred_class == 0) FN++;

    }   

    double precision = (TP + FP > 0) ? (double)TP / (TP + FP) : 0.0;
    double recall    = (TP + FN > 0) ? (double)TP / (TP + FN) : 0.0;
    double f1        = (precision + recall > 0) ? 
                    2 * precision * recall / (precision + recall) : 0.0;

    std::cout << "[ => Confusion Matrix:]"  << std::endl;
    std::cout << "[    TP=" << TP << " FP=" << FP << " ]" << std::endl;
    std::cout << "[    FN=" << FN << " TN=" << TN << " ]" << std::endl;
    std::cout << "[ => Precision : " << precision << " ]" << std::endl;;
    std::cout << "[ => Recall    : " << recall    << " ]" << std::endl;;
    std::cout << "[ => F1        : " << f1        << " ]" << std::endl;
*/
}



void NeuralNetwork::load_weights(const std::string &filename)
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

        this->weight_matrices[m] = Matrix(rows, cols, values);  
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

        this->bias_matrices[m] = Matrix(rows, cols, values);  
    }


    if(this->loss_function_class.weights.empty())
        this->loss_function_class.weights = Matrix::create_stacked_matrix(this->output_layer_neurons, 1,1,  1);

    this->lfunc = loss_function_class.get_fn(this->lfunc_type);
    this->lfunc_dx = loss_function_class.get_derivative_fn(this->lfunc_type, afunc_type.back());

    for(size_t a : afunc_type)
    {
        this->afunc.push_back(Activation::get_fn(a));
        this->afunc_dx.push_back(Activation::get_derivative_fn(a));
    }

    this->imported = true;
    std::cout << "[IMPORTED " << filename << " SUCCESSFULLY ]" << std::endl;

}

void NeuralNetwork::save_weights(const std::string &filename)
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


    for(Matrix &mat : this->weight_matrices )
    {
        file << mat.rows() << " " << mat.columns() << "\n";
        for(float value : mat.values() )
            file << value << " ";
        file << "\n";
    }

    for(Matrix &bias : this->bias_matrices )
    {
        file << bias.rows() << " " << bias.columns() << "\n";
        for(float value : bias.values() )
            file << value << " ";
        file << "\n";
    }

}
