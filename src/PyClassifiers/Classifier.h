#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <string>
#include <map>
#include <vector>

namespace pywrap {
    class Classifier {
    public:
        Classifier() = default;
        virtual ~Classifier() = default;
        virtual Classifier& fit(torch::Tensor& X, torch::Tensor& y, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states) = 0;
        virtual Classifier& fit(torch::Tensor& X, torch::Tensor& y) = 0;
        virtual torch::Tensor predict(torch::Tensor& X) = 0;
        virtual double score(torch::Tensor& X, torch::Tensor& y) = 0;
        virtual std::string version() = 0;
        virtual std::string sklearnVersion() = 0;
        virtual void setHyperparameters(const nlohmann::json& hyperparameters) = 0;
    protected:
        virtual void checkHyperparameters(const std::vector<std::string>& validKeys, const nlohmann::json& hyperparameters) = 0;
    };
} /* namespace pywrap */
#endif /* CLASSIFIER_H */