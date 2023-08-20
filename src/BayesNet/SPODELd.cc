#include "SPODELd.h"

namespace bayesnet {
    using namespace std;
    SPODELd::SPODELd(int root) : SPODE(root), Proposal(dataset, features, className) {}
    SPODELd& SPODELd::fit(torch::Tensor& X_, torch::Tensor& y_, vector<string>& features_, string className_, map<string, vector<int>>& states_)
    {
        // This first part should go in a Classifier method called fit_local_discretization o fit_float...
        features = features_;
        className = className_;
        Xf = X_;
        y = y_;
        // Fills vectors Xv & yv with the data from tensors X_ (discretized) & y
        states = fit_local_discretization(y);
        // We have discretized the input data
        // 1st we need to fit the model to build the normal SPODE structure, SPODE::fit initializes the base Bayesian network
        SPODE::fit(dataset, features, className, states);
        states = localDiscretizationProposal(states, model);
        return *this;
    }
    SPODELd& SPODELd::fit(torch::Tensor& dataset, vector<string>& features_, string className_, map<string, vector<int>>& states_)
    {
        Xf = dataset.index({ torch::indexing::Slice(0, dataset.size(0) - 1), "..." }).clone();
        y = dataset.index({ -1, "..." }).clone();
        // This first part should go in a Classifier method called fit_local_discretization o fit_float...
        features = features_;
        className = className_;
        // Fills vectors Xv & yv with the data from tensors X_ (discretized) & y
        states = fit_local_discretization(y);
        // We have discretized the input data
        // 1st we need to fit the model to build the normal SPODE structure, SPODE::fit initializes the base Bayesian network
        SPODE::fit(dataset, features, className, states);
        states = localDiscretizationProposal(states, model);
        return *this;
    }

    Tensor SPODELd::predict(Tensor& X)
    {
        auto Xt = prepareX(X);
        return SPODE::predict(Xt);
    }
    vector<string> SPODELd::graph(const string& name) const
    {
        return SPODE::graph(name);
    }
}