#include "Classifier.h"
#include "bayesnetUtils.h"

namespace bayesnet {
    using namespace torch;

    Classifier::Classifier(Network model) : model(model), m(0), n(0), metrics(Metrics()), fitted(false) {}
    Classifier& Classifier::build(const vector<string>& features, const string& className, map<string, vector<int>>& states, const torch::Tensor& weights)
    {
        this->features = features;
        this->className = className;
        this->states = states;
        m = dataset.size(1);
        n = dataset.size(0) - 1;
        checkFitParameters();
        auto n_classes = states[className].size();
        metrics = Metrics(dataset, features, className, n_classes);
        model.initialize();
        buildModel(weights);
        trainModel(weights);
        fitted = true;
        return *this;
    }
    void Classifier::buildDataset(Tensor& ytmp)
    {
        try {
            auto yresized = torch::transpose(ytmp.view({ ytmp.size(0), 1 }), 0, 1);
            dataset = torch::cat({ dataset, yresized }, 0);
        }
        catch (const std::exception& e) {
            std::cerr << e.what() << '\n';
            cout << "X dimensions: " << dataset.sizes() << "\n";
            cout << "y dimensions: " << ytmp.sizes() << "\n";
            exit(1);
        }
    }
    void Classifier::trainModel(const torch::Tensor& weights)
    {
        model.fit(dataset, weights, features, className, states);
    }
    // X is nxm where n is the number of features and m the number of samples
    Classifier& Classifier::fit(torch::Tensor& X, torch::Tensor& y, const vector<string>& features, const string& className, map<string, vector<int>>& states)
    {
        dataset = X;
        buildDataset(y);
        const torch::Tensor weights = torch::full({ dataset.size(1) }, 1.0 / dataset.size(1), torch::kDouble);
        return build(features, className, states, weights);
    }
    // X is nxm where n is the number of features and m the number of samples
    Classifier& Classifier::fit(vector<vector<int>>& X, vector<int>& y, const vector<string>& features, const string& className, map<string, vector<int>>& states)
    {
        dataset = torch::zeros({ static_cast<int>(X.size()), static_cast<int>(X[0].size()) }, kInt32);
        for (int i = 0; i < X.size(); ++i) {
            dataset.index_put_({ i, "..." }, torch::tensor(X[i], kInt32));
        }
        auto ytmp = torch::tensor(y, kInt32);
        buildDataset(ytmp);
        const torch::Tensor weights = torch::full({ dataset.size(1) }, 1.0 / dataset.size(1), torch::kDouble);
        return build(features, className, states, weights);
    }
    Classifier& Classifier::fit(torch::Tensor& dataset, const vector<string>& features, const string& className, map<string, vector<int>>& states)
    {
        this->dataset = dataset;
        const torch::Tensor weights = torch::full({ dataset.size(1) }, 1.0 / dataset.size(1), torch::kDouble);
        return build(features, className, states, weights);
    }
    Classifier& Classifier::fit(torch::Tensor& dataset, const vector<string>& features, const string& className, map<string, vector<int>>& states, const torch::Tensor& weights)
    {
        this->dataset = dataset;
        return build(features, className, states, weights);
    }
    void Classifier::checkFitParameters()
    {
        if (torch::is_floating_point(dataset)) {
            throw invalid_argument("dataset (X, y) must be of type Integer");
        }
        if (n != features.size()) {
            throw invalid_argument("X " + to_string(n) + " and features " + to_string(features.size()) + " must have the same number of features");
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
        return model.predict(X);
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
        Tensor y_pred = predict(X);
        return (y_pred == y).sum().item<float>() / y.size(0);
    }
    float Classifier::score(vector<vector<int>>& X, vector<int>& y)
    {
        if (!fitted) {
            throw logic_error("Classifier has not been fitted");
        }
        return model.score(X, y);
    }
    vector<string> Classifier::show() const
    {
        return model.show();
    }
    void Classifier::addNodes()
    {
        // Add all nodes to the network
        for (const auto& feature : features) {
            model.addNode(feature);
        }
        model.addNode(className);
    }
    int Classifier::getNumberOfNodes() const
    {
        // Features does not include class
        return fitted ? model.getFeatures().size() + 1 : 0;
    }
    int Classifier::getNumberOfEdges() const
    {
        return fitted ? model.getNumEdges() : 0;
    }
    int Classifier::getNumberOfStates() const
    {
        return fitted ? model.getStates() : 0;
    }
    vector<string> Classifier::topological_order()
    {
        return model.topological_sort();
    }
    void Classifier::dump_cpt() const
    {
        model.dump_cpt();
    }
    void Classifier::checkHyperparameters(const vector<string>& validKeys, nlohmann::json& hyperparameters)
    {
        for (const auto& item : hyperparameters.items()) {
            if (find(validKeys.begin(), validKeys.end(), item.key()) == validKeys.end()) {
                throw invalid_argument("Hyperparameter " + item.key() + " is not valid");
            }
        }
    }
    void Classifier::setHyperparameters(nlohmann::json& hyperparameters)
    {
        // Check if hyperparameters are valid, default is no hyperparameters
        const vector<string> validKeys = { };
        checkHyperparameters(validKeys, hyperparameters);
    }
}