// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include "SPODE.h"

namespace bayesnet {

    SPODE::SPODE(int root) : Classifier(Network()), root(root)
    {
        validHyperparameters = { "parent" };
    }

    void SPODE::setHyperparameters(const nlohmann::json& hyperparameters_)
    {
        auto hyperparameters = hyperparameters_;
        if (hyperparameters.contains("parent")) {
            root = hyperparameters["parent"];
            hyperparameters.erase("parent");
        }
        Classifier::setHyperparameters(hyperparameters);
    }
    void SPODE::buildModel(const torch::Tensor& weights)
    {
        // 0. Add all nodes to the model
        addNodes();
        // 1. Add edges from the class node to all other nodes
        // 2. Add edges from the root node to all other nodes
        if (root >= static_cast<int>(features.size())) {
            throw std::invalid_argument("The parent node is not in the dataset");
        }
        for (int i = 0; i < static_cast<int>(features.size()); ++i) {
            model.addEdge(className, features[i]);
            if (i != root) {
                model.addEdge(features[root], features[i]);
            }
        }
    }
    std::vector<std::string> SPODE::graph(const std::string& name) const
    {
        return model.graph(name);
    }

}