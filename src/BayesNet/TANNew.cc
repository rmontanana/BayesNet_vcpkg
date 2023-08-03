#include "TANNew.h"

namespace bayesnet {
    using namespace std;
    TANNew::TANNew() : TAN() {}
    TANNew::~TANNew() {}
    TANNew& TANNew::fit(torch::Tensor& X_, torch::Tensor& y_, vector<string>& features_, string className_, map<string, vector<int>>& states_)
    {
        Xf = X_;
        y = y_;
        features = features_;
        className = className_;
        Xv = vector<vector<int>>();
        yv = vector<int>(y.data_ptr<int>(), y.data_ptr<int>() + y.size(0));
        // discretize input data by feature(row)
        for (int i = 0; i < features.size(); ++i) {
            auto* discretizer = new mdlp::CPPFImdlp();
            auto Xt_ptr = Xf.index({ i }).data_ptr<float>();
            auto Xt = vector<float>(Xt_ptr, Xt_ptr + Xf.size(1));
            discretizer->fit(Xt, yv);
            Xv.push_back(discretizer->transform(Xt));
            auto xStates = vector<int>(discretizer->getCutPoints().size() + 1);
            iota(xStates.begin(), xStates.end(), 0);
            states[features[i]] = xStates;
            discretizers[features[i]] = discretizer;
        }
        int n_classes = torch::max(y).item<int>() + 1;
        auto yStates = vector<int>(n_classes);
        iota(yStates.begin(), yStates.end(), 0);
        states[className] = yStates;
        // Now we have standard TAN and now we implement the proposal
        // 1st we need to fit the model to build the TAN structure
        cout << "TANNew: Fitting model" << endl;
        TAN::fit(Xv, yv, features, className, states);
        cout << "TANNew: Model fitted" << endl;
        localDiscretizationProposal(discretizers, Xf);
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