#include "BoostAODE.h"

namespace bayesnet {
    BoostAODE::BoostAODE() : Ensemble() {}
    void BoostAODE::buildModel(const torch::Tensor& weights)
    {
        models.clear();
        for (int i = 0; i < features.size(); ++i) {
            models.push_back(std::make_unique<SPODE>(i));
        }
    }
    vector<string> BoostAODE::graph(const string& title) const
    {
        return Ensemble::graph(title);
    }
}