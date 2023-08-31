#include "Ensemble.h"

namespace bayesnet {
    using namespace torch;

    Ensemble::Ensemble() : Classifier(Network()), n_models(0) {}

    void Ensemble::trainModel(const torch::Tensor& weights)
    {
        n_models = models.size();
        for (auto i = 0; i < n_models; ++i) {
            // fit with vectors
            models[i]->fit(dataset, features, className, states);
        }
    }
    vector<int> Ensemble::voting(Tensor& y_pred)
    {
        auto y_pred_ = y_pred.accessor<int, 2>();
        vector<int> y_pred_final;
        for (int i = 0; i < y_pred.size(0); ++i) {
            vector<double> votes(y_pred.size(1), 0);
            for (int j = 0; j < y_pred.size(1); ++j) {
                votes[y_pred_[i][j]] += significanceModels[j];
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
        // //Create a threadpool
        // auto threads{ vector<thread>() };
        // mutex mtx;
        // for (auto i = 0; i < n_models; ++i) {
        //     threads.push_back(thread([&, i]() {
        //         auto ypredict = models[i]->predict(X);
        //         lock_guard<mutex> lock(mtx);
        //         y_pred.index_put_({ "...", i }, ypredict);
        //         }));
        //     Hacer voting aquí ? ? ?
        // }
        // for (auto& thread : threads) {
        //     thread.join();
        // }
        for (auto i = 0; i < n_models; ++i) {
            y_pred.index_put_({ "...", i }, models[i]->predict(X));
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
    vector<string> Ensemble::show() const
    {
        auto result = vector<string>();
        for (auto i = 0; i < n_models; ++i) {
            auto res = models[i]->show();
            result.insert(result.end(), res.begin(), res.end());
        }
        return result;
    }
    vector<string> Ensemble::graph(const string& title) const
    {
        auto result = vector<string>();
        for (auto i = 0; i < n_models; ++i) {
            auto res = models[i]->graph(title + "_" + to_string(i));
            result.insert(result.end(), res.begin(), res.end());
        }
        return result;
    }
    int Ensemble::getNumberOfNodes() const
    {
        int nodes = 0;
        for (auto i = 0; i < n_models; ++i) {
            nodes += models[i]->getNumberOfNodes();
        }
        return nodes;
    }
    int Ensemble::getNumberOfEdges() const
    {
        int edges = 0;
        for (auto i = 0; i < n_models; ++i) {
            edges += models[i]->getNumberOfEdges();
        }
        return edges;
    }
    int Ensemble::getNumberOfStates() const
    {
        int nstates = 0;
        for (auto i = 0; i < n_models; ++i) {
            nstates += models[i]->getNumberOfStates();
        }
        return nstates;
    }
}