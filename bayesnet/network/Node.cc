// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include "Node.h"

namespace bayesnet {

    Node::Node(const std::string& name)
        : name(name)
    {
    }
    void Node::clear()
    {
        parents.clear();
        children.clear();
        cpTable = torch::Tensor();
        dimensions.clear();
        numStates = 0;
    }
    std::string Node::getName() const
    {
        return name;
    }
    void Node::addParent(Node* parent)
    {
        parents.push_back(parent);
    }
    void Node::removeParent(Node* parent)
    {
        parents.erase(std::remove(parents.begin(), parents.end(), parent), parents.end());
    }
    void Node::removeChild(Node* child)
    {
        children.erase(std::remove(children.begin(), children.end(), child), children.end());
    }
    void Node::addChild(Node* child)
    {
        children.push_back(child);
    }
    std::vector<Node*>& Node::getParents()
    {
        return parents;
    }
    std::vector<Node*>& Node::getChildren()
    {
        return children;
    }
    int Node::getNumStates() const
    {
        return numStates;
    }
    void Node::setNumStates(int numStates)
    {
        this->numStates = numStates;
    }
    torch::Tensor& Node::getCPT()
    {
        return cpTable;
    }
    /*
     The MinFill criterion is a heuristic for variable elimination.
     The variable that minimizes the number of edges that need to be added to the graph to make it triangulated.
     This is done by counting the number of edges that need to be added to the graph if the variable is eliminated.
     The variable with the minimum number of edges is chosen.
     Here this is done computing the length of the combinations of the node neighbors taken 2 by 2.
    */
    unsigned Node::minFill()
    {
        std::unordered_set<std::string> neighbors;
        for (auto child : children) {
            neighbors.emplace(child->getName());
        }
        for (auto parent : parents) {
            neighbors.emplace(parent->getName());
        }
        auto source = std::vector<std::string>(neighbors.begin(), neighbors.end());
        return combinations(source).size();
    }
    std::vector<std::pair<std::string, std::string>> Node::combinations(const std::vector<std::string>& source)
    {
        std::vector<std::pair<std::string, std::string>> result;
        for (int i = 0; i < source.size(); ++i) {
            std::string temp = source[i];
            for (int j = i + 1; j < source.size(); ++j) {
                result.push_back({ temp, source[j] });
            }
        }
        return result;
    }
    void Node::computeCPT(const torch::Tensor& dataset, const std::vector<std::string>& features, const double smoothing, const torch::Tensor& weights)
    {
        dimensions.clear();
        dimensions.reserve(parents.size() + 1);
        // Get dimensions of the CPT
        dimensions.push_back(numStates);
        for (const auto& parent : parents) {
            dimensions.push_back(parent->getNumStates());
        }
        //transform(parents.begin(), parents.end(), back_inserter(dimensions), [](const auto& parent) { return parent->getNumStates(); });
        // Create a tensor initialized with smoothing
        cpTable = torch::full(dimensions, smoothing, torch::kDouble);
        // Create a map for quick feature index lookup
        std::unordered_map<std::string, int> featureIndexMap;
        for (size_t i = 0; i < features.size(); ++i) {
            featureIndexMap[features[i]] = i;
        }
        // Fill table with counts
        // Get the index of this node's feature
        int name_index = featureIndexMap[name];
        // Get parent indices in dataset
        std::vector<int> parent_indices;
        parent_indices.reserve(parents.size());
        for (const auto& parent : parents) {
            parent_indices.push_back(featureIndexMap[parent->getName()]);
        }
        c10::List<c10::optional<at::Tensor>> coordinates;
        for (int n_sample = 0; n_sample < dataset.size(1); ++n_sample) {
            coordinates.clear();
            auto sample = dataset.index({ "...", n_sample });
            coordinates.push_back(sample[name_index]);
            for (size_t i = 0; i < parent_indices.size(); ++i) {
                coordinates.push_back(sample[parent_indices[i]]);
            }
            // Increment the count of the corresponding coordinate
            cpTable.index_put_({ coordinates }, weights.index({ n_sample }), true);
        }
        // Normalize the counts (dividing each row by the sum of the row)
        cpTable /= cpTable.sum(0, true);
    }
    double Node::getFactorValue(std::map<std::string, int>& evidence)
    {
        c10::List<c10::optional<at::Tensor>> coordinates;
        // following predetermined order of indices in the cpTable (see Node.h)
        coordinates.push_back(at::tensor(evidence[name]));
        transform(parents.begin(), parents.end(), std::back_inserter(coordinates), [&evidence](const auto& parent) { return at::tensor(evidence[parent->getName()]); });
        return cpTable.index({ coordinates }).item<double>();
    }
    std::vector<std::string> Node::graph(const std::string& className)
    {
        auto output = std::vector<std::string>();
        auto suffix = name == className ? ", fontcolor=red, fillcolor=lightblue, style=filled " : "";
        output.push_back("\"" + name + "\" [shape=circle" + suffix + "] \n");
        transform(children.begin(), children.end(), back_inserter(output), [this](const auto& child) { return "\"" + name + "\" -> \"" + child->getName() + "\""; });
        return output;
    }
}