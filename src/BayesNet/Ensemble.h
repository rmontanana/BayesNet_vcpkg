#ifndef ENSEMBLE_H
#define ENSEMBLE_H
#include <torch/torch.h>
#include "Classifier.h"
#include "BayesMetrics.h"
#include "bayesnetUtils.h"
using namespace std;
using namespace torch;

namespace bayesnet {
    class Ensemble : public BaseClassifier {
    private:
        bool fitted;
        long n_models;
        Ensemble& build(vector<string>& features, string className, map<string, vector<int>>& states);
    protected:
        vector<unique_ptr<Classifier>> models;
        int m, n; // m: number of samples, n: number of features
        Tensor X;
        vector<vector<int>> Xv;
        Tensor y;
        vector<int> yv;
        Tensor dataset;
        Metrics metrics;
        vector<string> features;
        string className;
        map<string, vector<int>> states;
        void virtual train() = 0;
        vector<int> voting(Tensor& y_pred);
    public:
        Ensemble();
        virtual ~Ensemble() = default;
        Ensemble& fit(vector<vector<int>>& X, vector<int>& y, vector<string>& features, string className, map<string, vector<int>>& states) override;
        Ensemble& fit(torch::Tensor& X, torch::Tensor& y, vector<string>& features, string className, map<string, vector<int>>& states) override;
        Tensor predict(Tensor& X) override;
        vector<int> predict(vector<vector<int>>& X) override;
        float score(Tensor& X, Tensor& y) override;
        float score(vector<vector<int>>& X, vector<int>& y) override;
        int getNumberOfNodes() override;
        int getNumberOfEdges() override;
        int getNumberOfStates() override;
        vector<string> show() override;
        vector<string> graph(string title) override;
    };
}
#endif
