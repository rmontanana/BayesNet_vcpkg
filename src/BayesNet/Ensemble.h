#ifndef ENSEMBLE_H
#define ENSEMBLE_H
#include <torch/torch.h>
#include "Classifier.h"
#include "BayesMetrics.h"
#include "bayesnetUtils.h"
using namespace std;
using namespace torch;

namespace bayesnet {
    class Ensemble : public Classifier {
    private:
        Ensemble& build(vector<string>& features, string className, map<string, vector<int>>& states);
    protected:
        unsigned n_models;
        vector<unique_ptr<Classifier>> models;
        void trainModel(const torch::Tensor& weights) override;
        vector<int> voting(Tensor& y_pred);
    public:
        Ensemble();
        virtual ~Ensemble() = default;
        Tensor predict(Tensor& X) override;
        vector<int> predict(vector<vector<int>>& X) override;
        float score(Tensor& X, Tensor& y) override;
        float score(vector<vector<int>>& X, vector<int>& y) override;
        int getNumberOfNodes() const override;
        int getNumberOfEdges() const override;
        int getNumberOfStates() const override;
        vector<string> show() const override;
        vector<string> graph(const string& title) const override;
        vector<string> topological_order()  override
        {
            return vector<string>();
        }
        void dump_cpt() const override
        {
        }
    };
}
#endif
