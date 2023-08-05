#include "SPODENew.h"

namespace bayesnet {
    using namespace std;
    SPODENew::SPODENew(int root) : SPODE(root), Proposal(SPODE::Xv, SPODE::yv, features, className) {}
    SPODENew& SPODENew::fit(torch::Tensor& X_, torch::Tensor& y_, vector<string>& features_, string className_, map<string, vector<int>>& states_)
    {
        // This first part should go in a Classifier method called fit_local_discretization o fit_float...
        features = features_;
        className = className_;
        Xf = X_;
        y = y_;
        // Fills vectors Xv & yv with the data from tensors X_ (discretized) & y
        fit_local_discretization(states, y);
        generateTensorXFromVector();
        // We have discretized the input data
        // 1st we need to fit the model to build the normal SPODE structure, SPODE::fit initializes the base Bayesian network
        SPODE::fit(SPODE::Xv, SPODE::yv, features, className, states);
        localDiscretizationProposal(states, model);
        generateTensorXFromVector();
        Tensor ytmp = torch::transpose(y.view({ y.size(0), 1 }), 0, 1);
        samples = torch::cat({ X, ytmp }, 0);
        model.fit(SPODE::Xv, SPODE::yv, features, className);
        return *this;
    }
    Tensor SPODENew::predict(Tensor& X)
    {
        auto Xt = prepareX(X);
        return SPODE::predict(Xt);
    }
    vector<string> SPODENew::graph(const string& name)
    {
        return SPODE::graph(name);
    }
}