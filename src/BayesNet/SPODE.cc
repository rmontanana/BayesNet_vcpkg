#include "SPODE.h"

namespace bayesnet {

    SPODE::SPODE(int root) : Classifier(Network()), root(root) {}

    void SPODE::buildModel(const torch::Tensor& weights)
    {
        // 0. Add all nodes to the model
        addNodes();
        // 1. Add edges from the class node to all other nodes
        // 2. Add edges from the root node to all other nodes
        for (int i = 0; i < static_cast<int>(features.size()); ++i) {
            model.addEdge(className, features[i]);
            if (i != root) {
                model.addEdge(features[root], features[i]);
            }
        }
    }
    vector<string> SPODE::graph(const string& name) const
    {
        return model.graph(name);
    }

}