#include "Ensemble.h"

namespace bayesnet {
    using namespace std;
    using namespace torch;

    Ensemble::Ensemble(BaseClassifier& model) : model(model), models(vector<BaseClassifier>()), m(0), n(0), metrics(Metrics()) {}
    Ensemble& Ensemble::build(vector<string>& features, string className, map<string, vector<int>>& states)
    {

        dataset = torch::cat({ X, y.view({y.size(0), 1}) }, 1);
        this->features = features;
        this->className = className;
        this->states = states;
        auto n_classes = states[className].size();
        metrics = Metrics(dataset, features, className, n_classes);
        train();
        return *this;
    }
    Ensemble& Ensemble::fit(Tensor& X, Tensor& y, vector<string>& features, string className, map<string, vector<int>>& states)
    {
        this->X = X;
        this->y = y;
        auto sizes = X.sizes();
        m = sizes[0];
        n = sizes[1];
        return build(features, className, states);
    }
    Ensemble& Ensemble::fit(vector<vector<int>>& X, vector<int>& y, vector<string>& features, string className, map<string, vector<int>>& states)
    {
        this->X = torch::zeros({ static_cast<int64_t>(X[0].size()), static_cast<int64_t>(X.size()) }, kInt64);
        for (int i = 0; i < X.size(); ++i) {
            this->X.index_put_({ "...", i }, torch::tensor(X[i], kInt64));
        }
        this->y = torch::tensor(y, kInt64);
        return build(features, className, states);
    }
    Tensor Ensemble::predict(Tensor& X)
    {
        auto m_ = X.size(0);
        auto n_ = X.size(1);
        vector<vector<int>> Xd(n_, vector<int>(m_, 0));
        for (auto i = 0; i < n_; i++) {
            auto temp = X.index({ "...", i });
            Xd[i] = vector<int>(temp.data_ptr<int>(), temp.data_ptr<int>() + m_);
        }
        auto yp = model.predict(Xd);
        auto ypred = torch::tensor(yp, torch::kInt64);
        return ypred;
    }
    float Ensemble::score(Tensor& X, Tensor& y)
    {
        Tensor y_pred = predict(X);
        return (y_pred == y).sum().item<float>() / y.size(0);
    }
    vector<string> Ensemble::show()
    {
        return model.show();
    }

}