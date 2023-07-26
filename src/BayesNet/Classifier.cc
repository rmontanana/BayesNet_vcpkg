#include "Classifier.h"
#include "bayesnetUtils.h"

namespace bayesnet {
    using namespace torch;

    Classifier::Classifier(Network model) : model(model), m(0), n(0), metrics(Metrics()), fitted(false) {}
    Classifier& Classifier::build(vector<string>& features, string className, map<string, vector<int>>& states)
    {
        dataset = torch::cat({ X, y.view({y.size(0), 1}) }, 1);
        this->features = features;
        this->className = className;
        this->states = states;
        checkFitParameters();
        auto n_classes = states[className].size();
        metrics = Metrics(dataset, features, className, n_classes);
        train();
        if (Xv == vector<vector<int>>()) {
            // fit with tensors
            model.fit(X, y, features, className);
        } else {
            // fit with vectors
            model.fit(Xv, yv, features, className);
        }
        fitted = true;
        return *this;
    }
    Classifier& Classifier::fit(torch::Tensor& X, torch::Tensor& y, vector<string>& features, string className, map<string, vector<int>>& states)
    {
        this->X = torch::transpose(X, 0, 1);
        this->y = y;
        Xv = vector<vector<int>>();
        yv = vector<int>(y.data_ptr<int>(), y.data_ptr<int>() + y.size(0));
        return build(features, className, states);
    }

    Classifier& Classifier::fit(vector<vector<int>>& X, vector<int>& y, vector<string>& features, string className, map<string, vector<int>>& states)
    {
        this->X = torch::zeros({ static_cast<int>(X[0].size()), static_cast<int>(X.size()) }, kInt32);
        Xv = X;
        for (int i = 0; i < X.size(); ++i) {
            this->X.index_put_({ "...", i }, torch::tensor(X[i], kInt32));
        }
        this->y = torch::tensor(y, kInt32);
        yv = y;
        return build(features, className, states);
    }
    void Classifier::checkFitParameters()
    {
        auto sizes = X.sizes();
        m = sizes[0];
        n = sizes[1];
        if (m != y.size(0)) {
            throw invalid_argument("X and y must have the same number of samples");
        }
        if (n != features.size()) {
            throw invalid_argument("X and features must have the same number of features");
        }
        if (states.find(className) == states.end()) {
            throw invalid_argument("className not found in states");
        }
        for (auto feature : features) {
            if (states.find(feature) == states.end()) {
                throw invalid_argument("feature [" + feature + "] not found in states");
            }
        }
    }

    Tensor Classifier::predict(Tensor& X)
    {
        if (!fitted) {
            throw logic_error("Classifier has not been fitted");
        }
        auto m_ = X.size(0);
        auto n_ = X.size(1);
        //auto Xt = torch::transpose(X, 0, 1);
        vector<vector<int>> Xd(n_, vector<int>(m_, 0));
        for (auto i = 0; i < n_; i++) {
            auto temp = X.index({ "...", i });
            Xd[i] = vector<int>(temp.data_ptr<int>(), temp.data_ptr<int>() + temp.numel());
        }
        auto yp = model.predict(Xd);
        auto ypred = torch::tensor(yp, torch::kInt32);
        return ypred;
    }
    vector<int> Classifier::predict(vector<vector<int>>& X)
    {
        if (!fitted) {
            throw logic_error("Classifier has not been fitted");
        }
        auto m_ = X[0].size();
        auto n_ = X.size();
        vector<vector<int>> Xd(n_, vector<int>(m_, 0));
        for (auto i = 0; i < n_; i++) {
            Xd[i] = vector<int>(X[i].begin(), X[i].end());
        }
        auto yp = model.predict(Xd);
        return yp;
    }
    float Classifier::score(Tensor& X, Tensor& y)
    {
        if (!fitted) {
            throw logic_error("Classifier has not been fitted");
        }
        auto Xt = torch::transpose(X, 0, 1);
        Tensor y_pred = predict(Xt);
        return (y_pred == y).sum().item<float>() / y.size(0);
    }
    float Classifier::score(vector<vector<int>>& X, vector<int>& y)
    {
        if (!fitted) {
            throw logic_error("Classifier has not been fitted");
        }
        auto m_ = X[0].size();
        auto n_ = X.size();
        vector<vector<int>> Xd(n_, vector<int>(m_, 0));
        for (auto i = 0; i < n_; i++) {
            Xd[i] = vector<int>(X[i].begin(), X[i].end());
        }
        return model.score(Xd, y);
    }
    vector<string> Classifier::show()
    {
        return model.show();
    }
    void Classifier::addNodes()
    {
        auto test = model.getEdges();
        // Add all nodes to the network
        for (auto feature : features) {
            model.addNode(feature, states[feature].size());
        }
        model.addNode(className, states[className].size());
    }
    int Classifier::getNumberOfNodes()
    {
        // Features does not include class
        return fitted ? model.getFeatures().size() + 1 : 0;
    }
    int Classifier::getNumberOfEdges()
    {
        return fitted ? model.getEdges().size() : 0;
    }
}