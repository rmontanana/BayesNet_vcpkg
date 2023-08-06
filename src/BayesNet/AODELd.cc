#include "AODELd.h"

namespace bayesnet {
    using namespace std;
    AODELd::AODELd() : Ensemble(), Proposal(Ensemble::Xv, Ensemble::yv, features, className) {}
    AODELd& AODELd::fit(torch::Tensor& X_, torch::Tensor& y_, vector<string>& features_, string className_, map<string, vector<int>>& states_)
    {
        features = features_;
        className = className_;
        states = states_;
        train();
        for (const auto& model : models) {
            model->fit(X_, y_, features_, className_, states_);
        }
        n_models = models.size();
        fitted = true;
        return *this;
    }
    void AODELd::train()
    {
        models.clear();
        for (int i = 0; i < features.size(); ++i) {
            models.push_back(std::make_unique<SPODELd>(i));
        }
    }
    Tensor AODELd::predict(Tensor& X)
    {
        return Ensemble::predict(X);
    }
    vector<string> AODELd::graph(const string& name)
    {
        return Ensemble::graph(name);
    }
}