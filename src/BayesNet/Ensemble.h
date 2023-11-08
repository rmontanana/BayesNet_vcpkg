#ifndef ENSEMBLE_H
#define ENSEMBLE_H
#include <torch/torch.h>
#include "Classifier.h"
#include "BayesMetrics.h"
#include "bayesnetUtils.h"

namespace bayesnet {
    class Ensemble : public Classifier {
    private:
        Ensemble& build(std::vector<std::string>& features, std::string className, std::map<std::string, std::vector<int>>& states);
    protected:
        unsigned n_models;
        std::vector<std::unique_ptr<Classifier>> models;
        std::vector<double> significanceModels;
        void trainModel(const torch::Tensor& weights) override;
        std::vector<int> voting(torch::Tensor& y_pred);
    public:
        Ensemble();
        virtual ~Ensemble() = default;
        torch::Tensor predict(torch::Tensor& X) override;
        std::vector<int> predict(std::vector<std::vector<int>>& X) override;
        float score(torch::Tensor& X, torch::Tensor& y) override;
        float score(std::vector<std::vector<int>>& X, std::vector<int>& y) override;
        int getNumberOfNodes() const override;
        int getNumberOfEdges() const override;
        int getNumberOfStates() const override;
        std::vector<std::string> show() const override;
        std::vector<std::string> graph(const std::string& title) const override;
        std::vector<std::string> topological_order()  override
        {
            return std::vector<std::string>();
        }
        void dump_cpt() const override
        {
        }
    };
}
#endif
