#include "ODTE.h"

namespace pywrap {
    std::string ODTE::graph()
    {
        return callMethodString("graph");
    }
    void ODTE::setHyperparameters(const nlohmann::json& hyperparameters)
    {
        // Check if hyperparameters are valid
        const std::vector<std::string> validKeys = { "n_jobs", "n_estimators", "random_state" };
        checkHyperparameters(validKeys, hyperparameters);
        this->hyperparameters = hyperparameters;
    }
} /* namespace pywrap */