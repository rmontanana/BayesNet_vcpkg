#include "KDBLd.h"

namespace bayesnet {
    using namespace std;
    KDBLd::KDBLd(int k) : KDB(k), Proposal(dataset, features, className) {}
    KDBLd& KDBLd::fit(torch::Tensor& X_, torch::Tensor& y_, vector<string>& features_, string className_, map<string, vector<int>>& states_)
    {
        // This first part should go in a Classifier method called fit_local_discretization o fit_float...
        features = features_;
        className = className_;
        Xf = X_;
        y = y_;
        // Fills vectors Xv & yv with the data from tensors X_ (discretized) & y
        states = fit_local_discretization(y);
        // We have discretized the input data
        // 1st we need to fit the model to build the normal KDB structure, KDB::fit initializes the base Bayesian network
        KDB::fit(dataset, features, className, states);
        localDiscretizationProposal(states, model);
        return *this;
    }
    Tensor KDBLd::predict(Tensor& X)
    {
        auto Xt = prepareX(X);
        return KDB::predict(Xt);
    }
    vector<string> KDBLd::graph(const string& name) const
    {
        return KDB::graph(name);
    }
}