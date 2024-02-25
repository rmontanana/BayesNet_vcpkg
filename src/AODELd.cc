#include "AODELd.h"

namespace bayesnet {
    AODELd::AODELd(bool predict_voting) : Ensemble(predict_voting), Proposal(dataset, features, className)
    {
        validHyperparameters = { "predict_voting" };

    }
    void AODELd::setHyperparameters(const nlohmann::json& hyperparameters_)
    {
        auto hyperparameters = hyperparameters_;
        if (hyperparameters.contains("predict_voting")) {
            predict_voting = hyperparameters["predict_voting"];
            hyperparameters.erase("predict_voting");
        }
        if (!hyperparameters.empty()) {
            throw std::invalid_argument("Invalid hyperparameters" + hyperparameters.dump());
        }
    }
    AODELd& AODELd::fit(torch::Tensor& X_, torch::Tensor& y_, const std::vector<std::string>& features_, const std::string& className_, map<std::string, std::vector<int>>& states_)
    {
        checkInput(X_, y_);
        features = features_;
        className = className_;
        Xf = X_;
        y = y_;
        // Fills std::vectors Xv & yv with the data from tensors X_ (discretized) & y
        states = fit_local_discretization(y);
        // We have discretized the input data
        // 1st we need to fit the model to build the normal TAN structure, TAN::fit initializes the base Bayesian network
        Ensemble::fit(dataset, features, className, states);
        return *this;

    }
    void AODELd::buildModel(const torch::Tensor& weights)
    {
        models.clear();
        for (int i = 0; i < features.size(); ++i) {
            models.push_back(std::make_unique<SPODELd>(i));
        }
        n_models = models.size();
        significanceModels = std::vector<double>(n_models, 1.0);
    }
    void AODELd::trainModel(const torch::Tensor& weights)
    {
        for (const auto& model : models) {
            model->fit(Xf, y, features, className, states);
        }
    }
    std::vector<std::string> AODELd::graph(const std::string& name) const
    {
        return Ensemble::graph(name);
    }
}