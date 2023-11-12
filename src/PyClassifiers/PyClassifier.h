#ifndef PYCLASSIFIER_H
#define PYCLASSIFIER_H
#include "boost/python/detail/wrap_python.hpp"
#include <boost/python/numpy.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <map>
#include <vector>
#include <utility>
#include <torch/torch.h>
#include "PyWrap.h"
#include "Classifier.h"
#include "TypeId.h"

namespace pywrap {
    class PyClassifier : public Classifier {
    public:
        PyClassifier(const std::string& module, const std::string& className);
        virtual ~PyClassifier();
        PyClassifier& fit(torch::Tensor& X, torch::Tensor& y, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states) override;
        PyClassifier& fit(torch::Tensor& X, torch::Tensor& y) override;
        torch::Tensor predict(torch::Tensor& X) override;
        double score(torch::Tensor& X, torch::Tensor& y) override;
        std::string version() override;
        std::string sklearnVersion() override;
        std::string callMethodString(const std::string& method);
        void setHyperparameters(const nlohmann::json& hyperparameters) override;
    protected:
        void checkHyperparameters(const std::vector<std::string>& validKeys, const nlohmann::json& hyperparameters) override;
        nlohmann::json hyperparameters;
    private:
        PyWrap* pyWrap;
        std::string module;
        std::string className;
        clfId_t id;
        bool fitted;
    };
} /* namespace pywrap */
#endif /* PYCLASSIFIER_H */