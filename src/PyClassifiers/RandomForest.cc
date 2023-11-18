#include "RandomForest.h"

namespace pywrap {
    void RandomForest::setHyperparameters(const nlohmann::json& hyperparameters)
    {
        // Check if hyperparameters are valid
        const std::vector<std::string> validKeys = { "n_estimators", "n_jobs", "random_state" };
        checkHyperparameters(validKeys, hyperparameters);
        this->hyperparameters = hyperparameters;
    }
} /* namespace pywrap */