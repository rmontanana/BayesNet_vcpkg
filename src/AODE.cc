#include "AODE.h"

namespace bayesnet {
    AODE::AODE(bool predict_voting) : Ensemble(predict_voting)
    {
        validHyperparameters = { "predict_voting" };

    }
    void AODE::setHyperparameters(const nlohmann::json& hyperparameters_)
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
    void AODE::buildModel(const torch::Tensor& weights)
    {
        models.clear();
        significanceModels.clear();
        for (int i = 0; i < features.size(); ++i) {
            models.push_back(std::make_unique<SPODE>(i));
        }
        n_models = models.size();
        significanceModels = std::vector<double>(n_models, 1.0);
    }
    std::vector<std::string> AODE::graph(const std::string& title) const
    {
        return Ensemble::graph(title);
    }
}