#include "Factor.h"
#include <vector>
#include <string>

using namespace std;

namespace bayesnet {
    Factor::Factor(vector<string>& variables, vector<int>& cardinalities, torch::Tensor& values) : variables(variables), cardinalities(cardinalities), values(values) {}
    Factor::~Factor() = default;
    Factor::Factor(const Factor& other) : variables(other.variables), cardinalities(other.cardinalities), values(other.values) {}
    Factor& Factor::operator=(const Factor& other)
    {
        if (this != &other) {
            variables = other.variables;
            cardinalities = other.cardinalities;
            values = other.values;
        }
        return *this;
    }
    void Factor::setVariables(vector<string>& variables)
    {
        this->variables = variables;
    }
    void Factor::setCardinalities(vector<int>& cardinalities)
    {
        this->cardinalities = cardinalities;
    }
    void Factor::setValues(torch::Tensor& values)
    {
        this->values = values;
    }
    vector<string>& Factor::getVariables()
    {
        return variables;
    }
    vector<int>& Factor::getCardinalities()
    {
        return cardinalities;
    }
    torch::Tensor& Factor::getValues()
    {
        return values;
    }
    bool Factor::contains(string& variable)
    {
        for (int i = 0; i < variables.size(); i++) {
            if (variables[i] == variable) {
                return true;
            }
        }
        return false;
    }
    Factor* Factor::sumOut(string& candidate)
    {
        vector<string> newVariables;
        vector<int> newCardinalities;
        for (int i = 0; i < variables.size(); i++) {
            if (variables[i] != candidate) {
                newVariables.push_back(variables[i]);
                newCardinalities.push_back(cardinalities[i]);
            }
        }
        torch::Tensor newValues = values.sum(0);
        return new Factor(newVariables, newCardinalities, newValues);
    }
    Factor* Factor::product(vector<Factor>& factors)
    {
        vector<string> newVariables;
        vector<int> newCardinalities;
        for (auto factor : factors) {
            for (auto variable : factor.getVariables()) {
                if (find(newVariables.begin(), newVariables.end(), variable) == newVariables.end()) {
                    newVariables.push_back(variable);
                    newCardinalities.push_back(factor.getCardinalities()[factor.getVariables().index(variable)]);
                }
            }
        }
        torch::Tensor newValues = factors[0].getValues();
        for (int i = 1; i < factors.size(); i++) {
            newValues = newValues.matmul(factors[i].getValues());
        }
        return new Factor(newVariables, newCardinalities, newValues);
    }

}