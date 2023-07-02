#include "Factor.h"
#include <vector>
#include <string>

using namespace std;

namespace bayesnet {
    Factor::Factor(vector<string>& variables, vector<int>& cardinalities, torch::Tensor& values) : variables(variables), cardinalities(cardinalities), values(values) {}
    Factor::~Factor() = default;
}