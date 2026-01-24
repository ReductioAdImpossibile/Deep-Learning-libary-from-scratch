#include "model.h"
#include "matrix.h"
#include <string>

template<>
class model<CPU>
{
private:

public:

    void configure_input_layer(const size_t neurons);
    void add_hidden_layer(const size_t neurons, std::function<matrix<CPU>(matrix<CPU>)>);
    void confiugre_output_layer(const size_t neurons,  std::function<matrix<CPU>(matrix<CPU>)>);
    void load_csv(const std::string& filename, size_t label_col = 0, bool one_hot_output = false);

};
