#include "Ensemble.h"

namespace bayesnet {
    using namespace std;
    using namespace torch;

    Ensemble::Ensemble() : m(0), n(0), n_models(0), metrics(Metrics()) {}
    Ensemble& Ensemble::build(vector<string>& features, string className, map<string, vector<int>>& states)
    {
        dataset = torch::cat({ X, y.view({y.size(0), 1}) }, 1);
        this->features = features;
        this->className = className;
        this->states = states;
        auto n_classes = states[className].size();
        metrics = Metrics(dataset, features, className, n_classes);
        // Build models
        train();
        // Train models
        n_models = models.size();
        for (auto i = 0; i < n_models; ++i) {
            models[i].fit(X, y, features, className, states);
        }
        return *this;
    }
    Ensemble& Ensemble::fit(Tensor& X, Tensor& y, vector<string>& features, string className, map<string, vector<int>>& states)
    {
        this->X = X;
        this->y = y;
        auto sizes = X.sizes();
        m = sizes[0];
        n = sizes[1];
        return build(features, className, states);
    }
    Ensemble& Ensemble::fit(vector<vector<int>>& X, vector<int>& y, vector<string>& features, string className, map<string, vector<int>>& states)
    {
        this->X = torch::zeros({ static_cast<int64_t>(X[0].size()), static_cast<int64_t>(X.size()) }, kInt64);
        for (int i = 0; i < X.size(); ++i) {
            this->X.index_put_({ "...", i }, torch::tensor(X[i], kInt64));
        }
        this->y = torch::tensor(y, kInt64);
        return build(features, className, states);
    }
    Tensor Ensemble::predict(Tensor& X)
    {
        Tensor y_pred = torch::zeros({ X.size(0), n_models }, torch::kInt64);
        for (auto i = 0; i < n_models; ++i) {
            y_pred.index_put_({ "...", i }, models[i].predict(X));
        }
        auto y_pred_ = y_pred.accessor<int64_t, 2>();
        vector<int> y_pred_final;
        for (int i = 0; i < y_pred.size(0); ++i) {
            vector<float> votes(states[className].size(), 0);
            for (int j = 0; j < y_pred.size(1); ++j) {
                votes[y_pred_[i][j]] += 1;
            }
            auto indices = argsort(votes);
            y_pred_final.push_back(indices[0]);
        }
        return torch::tensor(y_pred_final, torch::kInt64);
    }
    float Ensemble::score(Tensor& X, Tensor& y)
    {
        Tensor y_pred = predict(X);
        return (y_pred == y).sum().item<float>() / y.size(0);
    }
    vector<string> Ensemble::show()
    {
        vector<string> result;
        for (auto i = 0; i < n_models; ++i) {
            auto res = models[i].show();
            result.insert(result.end(), res.begin(), res.end());
        }
        return result;
    }
}