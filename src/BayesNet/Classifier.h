#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <torch/torch.h>
#include "BaseClassifier.h"
#include "Network.h"
#include "BayesMetrics.h"

namespace bayesnet {
    class Classifier : public BaseClassifier {
    private:
        Classifier& build(const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const torch::Tensor& weights);
    protected:
        bool fitted;
        int m, n; // m: number of samples, n: number of features
        Network model;
        Metrics metrics;
        std::vector<std::string> features;
        std::string className;
        std::map<std::string, std::vector<int>> states;
        torch::Tensor dataset; // (n+1)xm tensor
        status_t status = NORMAL;
        void checkFitParameters();
        virtual void buildModel(const torch::Tensor& weights) = 0;
        void trainModel(const torch::Tensor& weights) override;
        void checkHyperparameters(const std::vector<std::string>& validKeys, nlohmann::json& hyperparameters);
        void buildDataset(torch::Tensor& y);
    public:
        Classifier(Network model);
        virtual ~Classifier() = default;
        Classifier& fit(std::vector<std::vector<int>>& X, std::vector<int>& y, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states) override;
        Classifier& fit(torch::Tensor& X, torch::Tensor& y, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states) override;
        Classifier& fit(torch::Tensor& dataset, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states) override;
        Classifier& fit(torch::Tensor& dataset, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const torch::Tensor& weights) override;
        void addNodes();
        int getNumberOfNodes() const override;
        int getNumberOfEdges() const override;
        int getNumberOfStates() const override;
        torch::Tensor predict(torch::Tensor& X) override;
        status_t getStatus() const override { return status; }
        std::vector<int> predict(std::vector<std::vector<int>>& X) override;
        float score(torch::Tensor& X, torch::Tensor& y) override;
        float score(std::vector<std::vector<int>>& X, std::vector<int>& y) override;
        std::vector<std::string> show() const override;
        std::vector<std::string> topological_order()  override;
        void dump_cpt() const override;
        void setHyperparameters(nlohmann::json& hyperparameters) override;
    };
}
#endif





