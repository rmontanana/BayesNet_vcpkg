// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef BOOSTA2DE_H
#define BOOSTA2DE_H
#include <map>
#include "boost.h"
#include "bayesnet/classifiers/SPnDE.h"
#include "bayesnet/feature_selection/FeatureSelect.h"
#include "Ensemble.h"
namespace bayesnet {
    class BoostA2DE : public Ensemble {
    public:
        explicit BoostA2DE(bool predict_voting = false);
        virtual ~BoostA2DE() = default;
        std::vector<std::string> graph(const std::string& title = "BoostA2DE") const override;
        void setHyperparameters(const nlohmann::json& hyperparameters_) override;
    protected:
        void buildModel(const torch::Tensor& weights) override;
    private:
        torch::Tensor X_train, y_train, X_test, y_test;
        // Hyperparameters
        bool bisection = true; // if true, use bisection stratety to add k models at once to the ensemble
        int maxTolerance = 3;
        std::string order_algorithm; // order to process the KBest features asc, desc, rand
        bool convergence = true; //if true, stop when the model does not improve
        bool convergence_best = false; // wether to keep the best accuracy to the moment or the last accuracy as prior accuracy
        bool selectFeatures = false; // if true, use feature selection
        std::string select_features_algorithm = Orders.DESC; // Selected feature selection algorithm
        FeatureSelect* featureSelector = nullptr;
        double threshold = -1;
        bool block_update = false;
    };
}
#endif