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

    BoostA2DE::BoostA2DE(bool predict_voting) : Boost(predict_voting)
    {
    }
    void BoostA2DE::buildModel(const torch::Tensor& weights)
    {
        // Models shall be built in trainModel
        models.clear();
        significanceModels.clear();
        n_models = 0;
        // Prepare the validation dataset
        auto y_ = dataset.index({ -1, "..." });
        if (convergence) {
            // Prepare train & validation sets from train data
            auto fold = folding::StratifiedKFold(5, y_, 271);
            auto [train, test] = fold.getFold(0);
            auto train_t = torch::tensor(train);
            auto test_t = torch::tensor(test);
            // Get train and validation sets
            X_train = dataset.index({ torch::indexing::Slice(0, dataset.size(0) - 1), train_t });
            y_train = dataset.index({ -1, train_t });
            X_test = dataset.index({ torch::indexing::Slice(0, dataset.size(0) - 1), test_t });
            y_test = dataset.index({ -1, test_t });
            dataset = X_train;
            m = X_train.size(1);
            auto n_classes = states.at(className).size();
            // Build dataset with train data
            buildDataset(y_train);
            metrics = Metrics(dataset, features, className, n_classes);
        } else {
            // Use all data to train
            X_train = dataset.index({ torch::indexing::Slice(0, dataset.size(0) - 1), "..." });
            y_train = y_;
        }

    }
    void BoostA2DE::trainModel(const torch::Tensor& weights)
    {

    }
    std::vector<std::string> BoostA2DE::graph(const std::string& title) const
    {
        return Ensemble::graph(title);
    }
}