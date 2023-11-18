#ifndef HYPERPARAMETERS_H
#define HYPERPARAMETERS_H
#include <string>
#include <map>
#include <vector>
#include <nlohmann/json.hpp>

namespace platform {
    using json = nlohmann::json;
    class HyperParameters {
    public:
        HyperParameters() = default;
        explicit HyperParameters(const std::vector<std::string>& datasets, const json& hyperparameters_);
        explicit HyperParameters(const std::vector<std::string>& datasets, const std::string& hyperparameters_file);
        ~HyperParameters() = default;
        bool notEmpty(const std::string& key) const { return hyperparameters.at(key) != json(); }
        json get(const std::string& key);
    private:
        std::map<std::string, json> hyperparameters;
    };
} /* namespace platform */
#endif /* HYPERPARAMETERS_H */