#include <string>
#include <vector>
#include <map>
#include "Network.h"

namespace bayesnet {
    Network::Network() {}

    Network::~Network()
    {
        for (auto& pair : nodes) {
            delete pair.second;
        }
    }

    void Network::addNode(std::string name)
    {
        nodes[name] = new Node(name);
    }

    void Network::addEdge(std::string parentName, std::string childName)
    {
        Node* parent = nodes[parentName];
        Node* child = nodes[childName];

        if (parent == nullptr || child == nullptr) {
            throw std::invalid_argument("Parent or child node not found.");
        }

        child->addParent(parent);
    }

    // to be implemented
    void Network::fit(const std::vector<std::vector<double>>& dataset)
    {
        // ... learn parameters (i.e., CPTs) using the dataset
    }

    // to be implemented
    std::vector<double> Network::predict(const std::vector<std::vector<double>>& testset)
    {
        std::vector<double> predictions;
        // ... use the CPTs and network structure to predict values
        return predictions;
    }
}

