// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include "A2DE.h"

namespace bayesnet {
    A2DE::A2DE(bool predict_voting) : Ensemble(predict_voting)
    {
        validHyperparameters = { "predict_voting" };

    }
    void A2DE::setHyperparameters(const nlohmann::json& hyperparameters_)
    {
        auto hyperparameters = hyperparameters_;
        if (hyperparameters.contains("predict_voting")) {
            predict_voting = hyperparameters["predict_voting"];
            hyperparameters.erase("predict_voting");
        }
        Classifier::setHyperparameters(hyperparameters);
    }
    void A2DE::buildModel(const torch::Tensor& weights)
    {
        models.clear();
        significanceModels.clear();
        for (int i = 0; i < features.size() - 1; ++i) {
            for (int j = i + 1; j < features.size(); ++j) {
                auto model = std::make_unique<SPnDE>(std::vector<int>({ i, j }));
                models.push_back(std::move(model));
            }
        }
        n_models = static_cast<unsigned>(models.size());
        significanceModels = std::vector<double>(n_models, 1.0);
    }
    std::vector<std::string> A2DE::graph(const std::string& title) const
    {
        return Ensemble::graph(title);
    }
}