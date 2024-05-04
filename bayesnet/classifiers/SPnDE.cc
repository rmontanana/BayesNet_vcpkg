// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include "SPnDE.h"

namespace bayesnet {

    SPnDE::SPnDE(std::vector<int> parents) : Classifier(Network()), parents(parents) {}

    void SPnDE::buildModel(const torch::Tensor& weights)
    {
        // 0. Add all nodes to the model
        addNodes();
        std::vector<int> attributes;
        for (int i = 0; i < static_cast<int>(features.size()); ++i) {
            if (std::find(parents.begin(), parents.end(), i) != parents.end()) {
                attributes.push_back(i);
            }
        }
        // 1. Add edges from the class node to all other nodes
        // 2. Add edges from the parents nodes to all other nodes
        for (const auto& attribute : attributes) {
            model.addEdge(className, features[attribute]);
            for (const auto& root : parents) {
                model.addEdge(features[root], features[attribute]);
            }
        }
    }
    std::vector<std::string> SPnDE::graph(const std::string& name) const
    {
        return model.graph(name);
    }

}