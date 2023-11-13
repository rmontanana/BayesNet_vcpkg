#include "SVC.h"

namespace pywrap {
    std::string SVC::version()
    {
        return sklearnVersion();
    }
    void SVC::setHyperparameters(nlohmann::json& hyperparameters)
    {
        // Check if hyperparameters are valid
        const std::vector<std::string> validKeys = { "C", "gamma", "kernel", "random_state" };
        checkHyperparameters(validKeys, hyperparameters);
        this->hyperparameters = hyperparameters;
    }
} /* namespace pywrap */