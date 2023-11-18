#include "SVC.h"

namespace pywrap {
    void SVC::setHyperparameters(const nlohmann::json& hyperparameters)
    {
        // Check if hyperparameters are valid
        const std::vector<std::string> validKeys = { "C", "gamma", "kernel", "random_state" };
        checkHyperparameters(validKeys, hyperparameters);
        this->hyperparameters = hyperparameters;
    }
} /* namespace pywrap */