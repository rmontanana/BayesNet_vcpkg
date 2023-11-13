#include "STree.h"

namespace pywrap {
    std::string STree::graph()
    {
        return callMethodString("graph");
    }
    void STree::setHyperparameters(nlohmann::json& hyperparameters)
    {
        // Check if hyperparameters are valid
        const std::vector<std::string> validKeys = { "C", "n_jobs", "kernel", "max_iter", "max_depth", "random_state", "multiclass_strategy" };
        checkHyperparameters(validKeys, hyperparameters);
        this->hyperparameters = hyperparameters;
    }
} /* namespace pywrap */