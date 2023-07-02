#ifndef FACTOR_H
#define FACTOR_H
#include <torch/torch.h>
#include <vector>
#include <string>
using namespace std;

namespace bayesnet {
    class Factor {
    private:
        vector<string> variables;
        vector<int> cardinalities;
        torch::Tensor values;
    public:
        Factor(vector<string>&, vector<int>&, torch::Tensor&);
        ~Factor();
        Factor(const Factor&);
        Factor& operator=(const Factor&);
        void setVariables(vector<string>&);
        void setCardinalities(vector<int>&);
        void setValues(torch::Tensor&);
        vector<string>& getVariables();
        vector<int>& getCardinalities();
        torch::Tensor& getValues();
    };
}
#endif