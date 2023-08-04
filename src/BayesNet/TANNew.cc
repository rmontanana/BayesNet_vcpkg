#include "TANNew.h"

namespace bayesnet {
    using namespace std;
    TANNew::TANNew() : TAN(), Proposal(TAN::Xv, TAN::yv, TAN::features, TAN::className) {}
    TANNew& TANNew::fit(torch::Tensor& X_, torch::Tensor& y_, vector<string>& features_, string className_, map<string, vector<int>>& states_)
    {
        // This first part should go in a Classifier method called fit_local_discretization o fit_float...
        TAN::features = features_;
        TAN::className = className_;
        Xf = X_;
        y = y_;
        fit_local_discretization(states, y);
        // We have discretized the input data
        // 1st we need to fit the model to build the normal TAN structure, TAN::fit initializes the base Bayesian network
        cout << "TANNew: Fitting model" << endl;
        TAN::fit(TAN::Xv, TAN::yv, TAN::features, TAN::className, states);
        cout << "TANNew: Model fitted" << endl;
        localDiscretizationProposal(states, model);
        addNodes();
        model.fit(TAN::Xv, TAN::yv, features, className);
        return *this;
    }
    Tensor TANNew::predict(Tensor& X)
    {
        auto Xtd = torch::zeros_like(X, torch::kInt32);
        for (int i = 0; i < X.size(0); ++i) {
            auto Xt = vector<float>(X[i].data_ptr<float>(), X[i].data_ptr<float>() + X.size(1));
            auto Xd = discretizers[features[i]]->transform(Xt);
            Xtd.index_put_({ i }, torch::tensor(Xd, torch::kInt32));
        }
        cout << "TANNew Xtd: " << Xtd.sizes() << endl;
        return TAN::predict(Xtd);
    }
    vector<string> TANNew::graph(const string& name)
    {
        return TAN::graph(name);
    }
}