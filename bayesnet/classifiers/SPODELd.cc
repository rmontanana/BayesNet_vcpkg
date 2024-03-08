#include "SPODELd.h"

namespace bayesnet {
    SPODELd::SPODELd(int root) : SPODE(root), Proposal(dataset, features, className) {}
    SPODELd& SPODELd::fit(torch::Tensor& X_, torch::Tensor& y_, const std::vector<std::string>& features_, const std::string& className_, map<std::string, std::vector<int>>& states_)
    {
        checkInput(X_, y_);
        features = features_;
        className = className_;
        Xf = X_;
        y = y_;
        // Fills std::vectors Xv & yv with the data from tensors X_ (discretized) & y
        states = fit_local_discretization(y);
        // We have discretized the input data
        // 1st we need to fit the model to build the normal SPODE structure, SPODE::fit initializes the base Bayesian network
        SPODE::fit(dataset, features, className, states);
        states = localDiscretizationProposal(states, model);
        return *this;
    }
    SPODELd& SPODELd::fit(torch::Tensor& dataset, const std::vector<std::string>& features_, const std::string& className_, map<std::string, std::vector<int>>& states_)
    {
        if (!torch::is_floating_point(dataset)) {
            throw std::runtime_error("Dataset must be a floating point tensor");
        }
        Xf = dataset.index({ torch::indexing::Slice(0, dataset.size(0) - 1), "..." }).clone();
        y = dataset.index({ -1, "..." }).clone();
        features = features_;
        className = className_;
        // Fills std::vectors Xv & yv with the data from tensors X_ (discretized) & y
        states = fit_local_discretization(y);
        // We have discretized the input data
        // 1st we need to fit the model to build the normal SPODE structure, SPODE::fit initializes the base Bayesian network
        SPODE::fit(dataset, features, className, states);
        states = localDiscretizationProposal(states, model);
        return *this;
    }

    torch::Tensor SPODELd::predict(torch::Tensor& X)
    {
        auto Xt = prepareX(X);
        return SPODE::predict(Xt);
    }
    std::vector<std::string> SPODELd::graph(const std::string& name) const
    {
        return SPODE::graph(name);
    }
}