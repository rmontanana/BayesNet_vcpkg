#include "TANNew.h"

namespace bayesnet {
    using namespace std;
    TANNew::TANNew() : TAN(), n_features{ 0 } {}
    TANNew::~TANNew() {}
    TANNew& TANNew::fit(torch::Tensor& X, torch::Tensor& y, vector<string>& features, string className, map<string, vector<int>>& states)
    {
        n_features = features.size();
        this->Xf = torch::transpose(X, 0, 1); // now it is mxn as X comes in nxm
        this->y = y;
        this->features = features;
        this->className = className;
        Xv = vector<vector<int>>();
        yv = vector<int>(y.data_ptr<int>(), y.data_ptr<int>() + y.size(0));
        for (int i = 0; i < features.size(); ++i) {
            auto* discretizer = new mdlp::CPPFImdlp();
            auto Xt_ptr = X.index({ i }).data_ptr<float>();
            auto Xt = vector<float>(Xt_ptr, Xt_ptr + X.size(1));
            discretizer->fit(Xt, yv);
            Xv.push_back(discretizer->transform(Xt));
            auto xStates = vector<int>(discretizer->getCutPoints().size() + 1);
            iota(xStates.begin(), xStates.end(), 0);
            this->states[features[i]] = xStates;
            discretizers[features[i]] = discretizer;
        }
        int n_classes = torch::max(y).item<int>() + 1;
        auto yStates = vector<int>(n_classes);
        iota(yStates.begin(), yStates.end(), 0);
        this->states[className] = yStates;
        /*
        Hay que discretizar los datos de entrada y luego en predict discretizar también con el mmismo modelo, hacer un transform solamente.
        */
        TAN::fit(Xv, yv, features, className, this->states);
        return *this;
    }
    void TANNew::train()
    {
        TAN::train();
    }
    Tensor TANNew::predict(Tensor& X)
    {
        auto Xtd = torch::zeros_like(X, torch::kInt32);
        for (int i = 0; i < X.size(0); ++i) {
            auto Xt = vector<float>(X[i].data_ptr<float>(), X[i].data_ptr<float>() + X.size(1));
            auto Xd = discretizers[features[i]]->transform(Xt);
            Xtd.index_put_({ i }, torch::tensor(Xd, torch::kInt32));
        }
        return TAN::predict(Xtd);
    }
    vector<string> TANNew::graph(const string& name)
    {
        return TAN::graph(name);
    }
}