#ifndef ENSEMBLE_H
#define ENSEMBLE_H
#include <torch/torch.h>
#include "Classifier.h"
#include "BayesMetrics.h"
#include "bayesnetUtils.h"

namespace bayesnet {
    class Ensemble : public Classifier {
    public:
        Ensemble(bool predict_voting = true);
        virtual ~Ensemble() = default;
        torch::Tensor predict(torch::Tensor& X) override;
        std::vector<int> predict(std::vector<std::vector<int>>& X) override;
        torch::Tensor predict_proba(torch::Tensor& X) override;
        std::vector<std::vector<double>> predict_proba(std::vector<std::vector<int>>& X) override;
        torch::Tensor do_predict_voting(torch::Tensor& X);
        std::vector<int> do_predict_voting(std::vector<std::vector<int>>& X);
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
    protected:
        unsigned n_models;
        std::vector<std::unique_ptr<Classifier>> models;
        std::vector<double> significanceModels;
        void trainModel(const torch::Tensor& weights) override;
        std::vector<int> voting(torch::Tensor& y_pred);
    private:
        bool predict_voting;
    };
}
#endif
