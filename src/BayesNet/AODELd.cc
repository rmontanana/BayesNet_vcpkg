#include "AODELd.h"
#include "Models.h"

namespace bayesnet {
    using namespace std;
    AODELd::AODELd() : Ensemble(), Proposal(dataset, features, className) {}
    AODELd& AODELd::fit(torch::Tensor& X_, torch::Tensor& y_, const vector<string>& features_, const string& className_, map<string, vector<int>>& states_)
    {
        checkInput(X_, y_);
        features = features_;
        className = className_;
        Xf = X_;
        y = y_;
        // Fills vectors Xv & yv with the data from tensors X_ (discretized) & y
        states = fit_local_discretization(y);
        // We have discretized the input data
        // 1st we need to fit the model to build the normal TAN structure, TAN::fit initializes the base Bayesian network
        Ensemble::fit(dataset, features, className, states);
        return *this;

    }
    void AODELd::buildModel(const torch::Tensor& weights)
    {
        models.clear();
        for (int i = 0; i < features.size(); ++i) {
            models.push_back(std::make_unique<SPODELd>(i));
        }
        n_models = models.size();
    }
    void AODELd::trainModel(const torch::Tensor& weights)
    {
        for (const auto& model : models) {
            model->fit(Xf, y, features, className, states);
        }
    }
    vector<string> AODELd::graph(const string& name) const
    {
        return Ensemble::graph(name);
    }
}