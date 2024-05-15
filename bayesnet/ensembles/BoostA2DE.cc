// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <set>
#include <functional>
#include <limits.h>
#include <tuple>
#include <folding.hpp>
#include "bayesnet/feature_selection/CFS.h"
#include "bayesnet/feature_selection/FCBF.h"
#include "bayesnet/feature_selection/IWSS.h"
#include "BoostA2DE.h"

namespace bayesnet {

    BoostA2DE::BoostA2DE(bool predict_voting) : Ensemble(predict_voting)
    {
        validHyperparameters = {
            "maxModels", "bisection", "order", "convergence", "convergence_best", "threshold",
            "select_features", "maxTolerance", "predict_voting", "block_update"
        };

    }
    void BoostA2DE::buildModel(const torch::Tensor& weights)
    {
        models.clear();

    }
    void BoostA2DE::setHyperparameters(const nlohmann::json& hyperparameters_)
    {
        auto hyperparameters = hyperparameters_;
        if (hyperparameters.contains("order")) {
            std::vector<std::string> algos = { Orders.ASC, Orders.DESC, Orders.RAND };
            order_algorithm = hyperparameters["order"];
            if (std::find(algos.begin(), algos.end(), order_algorithm) == algos.end()) {
                throw std::invalid_argument("Invalid order algorithm, valid values [" + Orders.ASC + ", " + Orders.DESC + ", " + Orders.RAND + "]");
            }
            hyperparameters.erase("order");
        }
        if (hyperparameters.contains("convergence")) {
            convergence = hyperparameters["convergence"];
            hyperparameters.erase("convergence");
        }
        if (hyperparameters.contains("convergence_best")) {
            convergence_best = hyperparameters["convergence_best"];
            hyperparameters.erase("convergence_best");
        }
        if (hyperparameters.contains("bisection")) {
            bisection = hyperparameters["bisection"];
            hyperparameters.erase("bisection");
        }
        if (hyperparameters.contains("threshold")) {
            threshold = hyperparameters["threshold"];
            hyperparameters.erase("threshold");
        }
        if (hyperparameters.contains("maxTolerance")) {
            maxTolerance = hyperparameters["maxTolerance"];
            if (maxTolerance < 1 || maxTolerance > 4)
                throw std::invalid_argument("Invalid maxTolerance value, must be greater in [1, 4]");
            hyperparameters.erase("maxTolerance");
        }
        if (hyperparameters.contains("predict_voting")) {
            predict_voting = hyperparameters["predict_voting"];
            hyperparameters.erase("predict_voting");
        }
        if (hyperparameters.contains("select_features")) {
            auto selectedAlgorithm = hyperparameters["select_features"];
            std::vector<std::string> algos = { SelectFeatures.IWSS, SelectFeatures.CFS, SelectFeatures.FCBF };
            selectFeatures = true;
            select_features_algorithm = selectedAlgorithm;
            if (std::find(algos.begin(), algos.end(), selectedAlgorithm) == algos.end()) {
                throw std::invalid_argument("Invalid selectFeatures value, valid values [" + SelectFeatures.IWSS + ", " + SelectFeatures.CFS + ", " + SelectFeatures.FCBF + "]");
            }
            hyperparameters.erase("select_features");
        }
        if (hyperparameters.contains("block_update")) {
            block_update = hyperparameters["block_update"];
            hyperparameters.erase("block_update");
        }
        Classifier::setHyperparameters(hyperparameters);
    }
    std::vector<std::string> BoostA2DE::graph(const std::string& title) const
    {
        return Ensemble::graph(title);
    }
}