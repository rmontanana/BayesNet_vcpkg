#include "HyperParameters.h"
#include <fstream>

namespace platform {
    HyperParameters::HyperParameters(const std::vector<std::string>& datasets, const json& hyperparameters_)
    {
        // Initialize all datasets with the given hyperparameters
        for (const auto& item : datasets) {
            hyperparameters[item] = hyperparameters_;
        }
    }
    HyperParameters::HyperParameters(const std::vector<std::string>& datasets, const std::string& hyperparameters_file)
    {
        // Check if file exists
        std::ifstream file(hyperparameters_file);
        if (!file.is_open()) {
            throw std::runtime_error("File " + hyperparameters_file + " not found");
        }
        // Check if file is a json
        json input_hyperparameters = json::parse(file);
        // Check if hyperparameters are valid
        for (const auto& dataset : datasets) {
            if (!input_hyperparameters.contains(dataset)) {
                throw std::runtime_error("Dataset " + dataset + " not found in hyperparameters file");
            }
            hyperparameters[dataset] = input_hyperparameters[dataset];
        }
    }
    json HyperParameters::get(const std::string& key)
    {
        return hyperparameters.at(key);
    }
} /* namespace platform */