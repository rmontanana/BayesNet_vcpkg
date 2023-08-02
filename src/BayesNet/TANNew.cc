#include "TANNew.h"
#include "ArffFiles.h"

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
        auto Xvf = vector<vector<float>>();
        yv = vector<int>(y.data_ptr<int>(), y.data_ptr<int>() + y.size(0));
        for (int i = 0; i < features.size(); ++i) {
            auto* discretizer = new mdlp::CPPFImdlp();
            auto Xt_ptr = X.index({ i }).data_ptr<float>();
            auto Xt = vector<float>(Xt_ptr, Xt_ptr + X.size(1));
            Xvf.push_back(Xt);
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
        // Now we have standard TAN and now we implement the proposal
        // order of local discretization is important. no good 0, 1, 2...
        auto order = model.topological_sort();
        auto nodes = model.getNodes();
        bool upgrade = false; // Flag to check if we need to upgrade the model
        for (auto feature : order) {
            auto nodeParents = nodes[feature]->getParents();
            int index = find(features.begin(), features.end(), feature) - features.begin();
            vector<string> parents;
            if (parents.size() == 1) continue; // Only has class as parent
            upgrade = true;
            transform(nodeParents.begin(), nodeParents.end(), back_inserter(parents), [](const auto& p) {return p->getName(); });
            // Remove class as parent as it will be added later
            parents.erase(remove(parents.begin(), parents.end(), className), parents.end());
            vector<int> indices;
            transform(parents.begin(), parents.end(), back_inserter(indices), [&](const auto& p) {return find(features.begin(), features.end(), p) - features.begin(); });
            // Now we fit the discretizer of the feature conditioned on its parents and the class i.e. discretizer.fit(X[index], X[indices] + y)
            vector<string> yJoinParents;
            transform(yv.begin(), yv.end(), back_inserter(yJoinParents), [&](const auto& p) {return to_string(p); });
            for (auto idx : indices) {
                for (int i = 0; i < Xvf[idx].size(); ++i) {
                    yJoinParents[i] += to_string(Xv[idx][i]);
                }
            }
            auto arff = ArffFiles();
            auto yxv = arff.factorize(yJoinParents);
            discretizers[feature]->fit(Xvf[index], yxv);
        }
        if (upgrade) {
            // Discretize again X with the new fitted discretizers
            Xv = vector<vector<int>>();
            for (int i = 0; i < features.size(); ++i) {
                auto Xt_ptr = X.index({ i }).data_ptr<float>();
                auto Xt = vector<float>(Xt_ptr, Xt_ptr + X.size(1));
                Xv.push_back(discretizers[features[i]]->transform(Xt));
                auto xStates = vector<int>(discretizers[features[i]]->getCutPoints().size() + 1);
                iota(xStates.begin(), xStates.end(), 0);
                this->states[features[i]] = xStates;
            }
        }


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