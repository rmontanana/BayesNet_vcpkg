#include "AODELd.h"

namespace bayesnet {
    using namespace std;
    AODELd::AODELd() : Ensemble(), Proposal(dataset, features, className) {}
    AODELd& AODELd::fit(torch::Tensor& X_, torch::Tensor& y_, vector<string>& features_, string className_, map<string, vector<int>>& states_)
    {
        features = features_;
        className = className_;
        states = states_;
        buildModel();
        trainModel();
        n_models = models.size();
        fitted = true;
        return *this;
    }
    void AODELd::buildModel()
    {
        models.clear();
        for (int i = 0; i < features.size(); ++i) {
            models.push_back(std::make_unique<SPODELd>(i));
        }
    }
    void AODELd::trainModel()
    {
        for (const auto& model : models) {
            model->fit(dataset, features, className, states);
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