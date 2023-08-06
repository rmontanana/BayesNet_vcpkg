#include "Ensemble.h"

namespace bayesnet {
    using namespace torch;

    Ensemble::Ensemble() : n_models(0), metrics(Metrics()), fitted(false) {}
    Ensemble& Ensemble::build(vector<string>& features, string className, map<string, vector<int>>& states)
    {
        Tensor ytmp = torch::transpose(y.view({ y.size(0), 1 }), 0, 1);
        samples = torch::cat({ X, ytmp }, 0);
        this->features = features;
        this->className = className;
        this->states = states;
        auto n_classes = states[className].size();
        metrics = Metrics(samples, features, className, n_classes);
        // Build models
        train();
        // Train models
        n_models = models.size();
        for (auto i = 0; i < n_models; ++i) {
            if (Xv.empty()) {
                // fit with tensors
                models[i]->fit(X, y, features, className, states);
            } else {
                // fit with vectors
                models[i]->fit(Xv, yv, features, className, states);
            }
        }
        fitted = true;
        return *this;
    }
    void Ensemble::generateTensorXFromVector()
    {
        X = torch::zeros({ static_cast<int>(Xv.size()), static_cast<int>(Xv[0].size()) }, kInt32);
        for (int i = 0; i < Xv.size(); ++i) {
            X.index_put_({ i, "..." }, torch::tensor(Xv[i], kInt32));
        }
    }
    Ensemble& Ensemble::fit(torch::Tensor& X, torch::Tensor& y, vector<string>& features, string className, map<string, vector<int>>& states)
    {
        this->X = X;
        this->y = y;
        Xv = vector<vector<int>>();
        yv = vector<int>(y.data_ptr<int>(), y.data_ptr<int>() + y.size(0));
        return build(features, className, states);
    }
    Ensemble& Ensemble::fit(vector<vector<int>>& X, vector<int>& y, vector<string>& features, string className, map<string, vector<int>>& states)
    {
        Xv = X;
        generateTensorXFromVector();
        this->y = torch::tensor(y, kInt32);
        yv = y;
        return build(features, className, states);
    }
    vector<int> Ensemble::voting(Tensor& y_pred)
    {
        auto y_pred_ = y_pred.accessor<int, 2>();
        vector<int> y_pred_final;
        for (int i = 0; i < y_pred.size(0); ++i) {
            vector<float> votes(y_pred.size(1), 0);
            for (int j = 0; j < y_pred.size(1); ++j) {
                votes[y_pred_[i][j]] += 1;
            }
            // argsort in descending order
            auto indices = argsort(votes);
            y_pred_final.push_back(indices[0]);
        }
        return y_pred_final;
    }
    Tensor Ensemble::predict(Tensor& X)
    {
        if (!fitted) {
            throw logic_error("Ensemble has not been fitted");
        }
        Tensor y_pred = torch::zeros({ X.size(1), n_models }, kInt32);
        //Create a threadpool
        auto threads{ vector<thread>() };
        mutex mtx;
        for (auto i = 0; i < n_models; ++i) {
            threads.push_back(thread([&, i]() {
                auto ypredict = models[i]->predict(X);
                lock_guard<mutex> lock(mtx);
                y_pred.index_put_({ "...", i }, ypredict);
                }));
        }
        for (auto& thread : threads) {
            thread.join();
        }
        return torch::tensor(voting(y_pred));
    }
    vector<int> Ensemble::predict(vector<vector<int>>& X)
    {
        if (!fitted) {
            throw logic_error("Ensemble has not been fitted");
        }
        long m_ = X[0].size();
        long n_ = X.size();
        vector<vector<int>> Xd(n_, vector<int>(m_, 0));
        for (auto i = 0; i < n_; i++) {
            Xd[i] = vector<int>(X[i].begin(), X[i].end());
        }
        Tensor y_pred = torch::zeros({ m_, n_models }, kInt32);
        for (auto i = 0; i < n_models; ++i) {
            y_pred.index_put_({ "...", i }, torch::tensor(models[i]->predict(Xd), kInt32));
        }
        return voting(y_pred);
    }
    float Ensemble::score(Tensor& X, Tensor& y)
    {
        if (!fitted) {
            throw logic_error("Ensemble has not been fitted");
        }
        auto y_pred = predict(X);
        int correct = 0;
        for (int i = 0; i < y_pred.size(0); ++i) {
            if (y_pred[i].item<int>() == y[i].item<int>()) {
                correct++;
            }
        }
        return (double)correct / y_pred.size(0);
    }
    float Ensemble::score(vector<vector<int>>& X, vector<int>& y)
    {
        if (!fitted) {
            throw logic_error("Ensemble has not been fitted");
        }
        auto y_pred = predict(X);
        int correct = 0;
        for (int i = 0; i < y_pred.size(); ++i) {
            if (y_pred[i] == y[i]) {
                correct++;
            }
        }
        return (double)correct / y_pred.size();

    }
    vector<string> Ensemble::show()
    {
        auto result = vector<string>();
        for (auto i = 0; i < n_models; ++i) {
            auto res = models[i]->show();
            result.insert(result.end(), res.begin(), res.end());
        }
        return result;
    }
    vector<string> Ensemble::graph(const string& title)
    {
        auto result = vector<string>();
        for (auto i = 0; i < n_models; ++i) {
            auto res = models[i]->graph(title + "_" + to_string(i));
            result.insert(result.end(), res.begin(), res.end());
        }
        return result;
    }
    int Ensemble::getNumberOfNodes()
    {
        int nodes = 0;
        for (auto i = 0; i < n_models; ++i) {
            nodes += models[i]->getNumberOfNodes();
        }
        return nodes;
    }
    int Ensemble::getNumberOfEdges()
    {
        int edges = 0;
        for (auto i = 0; i < n_models; ++i) {
            edges += models[i]->getNumberOfEdges();
        }
        return edges;
    }
    int Ensemble::getNumberOfStates()
    {
        int nstates = 0;
        for (auto i = 0; i < n_models; ++i) {
            nstates += models[i]->getNumberOfStates();
        }
        return nstates;
    }
}