#include "Node.h"

namespace bayesnet {

    Node::Node(const std::string& name)
        : name(name), numStates(0), cpTable(torch::Tensor()), parents(vector<Node*>()), children(vector<Node*>())
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
    string Node::getName() const
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
    vector<Node*>& Node::getParents()
    {
        return parents;
    }
    vector<Node*>& Node::getChildren()
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
        unordered_set<string> neighbors;
        for (auto child : children) {
            neighbors.emplace(child->getName());
        }
        for (auto parent : parents) {
            neighbors.emplace(parent->getName());
        }
        auto source = vector<string>(neighbors.begin(), neighbors.end());
        return combinations(source).size();
    }
    vector<pair<string, string>> Node::combinations(const vector<string>& source)
    {
        vector<pair<string, string>> result;
        for (int i = 0; i < source.size(); ++i) {
            string temp = source[i];
            for (int j = i + 1; j < source.size(); ++j) {
                result.push_back({ temp, source[j] });
            }
        }
        return result;
    }
    void Node::computeCPT(const torch::Tensor& dataset, const vector<string>& features, const double laplaceSmoothing, const torch::Tensor& weights)
    {
        dimensions.clear();
        // Get dimensions of the CPT
        dimensions.push_back(numStates);
        transform(parents.begin(), parents.end(), back_inserter(dimensions), [](const auto& parent) { return parent->getNumStates(); });

        // Create a tensor of zeros with the dimensions of the CPT
        cpTable = torch::zeros(dimensions, torch::kFloat) + laplaceSmoothing;
        // Fill table with counts
        auto pos = find(features.begin(), features.end(), name);
        if (pos == features.end()) {
            throw logic_error("Feature " + name + " not found in dataset");
        }
        int name_index = pos - features.begin();
        for (int n_sample = 0; n_sample < dataset.size(1); ++n_sample) {
            c10::List<c10::optional<at::Tensor>> coordinates;
            coordinates.push_back(dataset.index({ name_index, n_sample }));
            for (auto parent : parents) {
                pos = find(features.begin(), features.end(), parent->getName());
                if (pos == features.end()) {
                    throw logic_error("Feature parent " + parent->getName() + " not found in dataset");
                }
                int parent_index = pos - features.begin();
                coordinates.push_back(dataset.index({ parent_index, n_sample }));
            }
            // Increment the count of the corresponding coordinate
            cpTable.index_put_({ coordinates }, cpTable.index({ coordinates }) + weights.index({ n_sample }).item<double>());
        }
        // Normalize the counts
        cpTable = cpTable / cpTable.sum(0);
    }
    float Node::getFactorValue(map<string, int>& evidence)
    {
        c10::List<c10::optional<at::Tensor>> coordinates;
        // following predetermined order of indices in the cpTable (see Node.h)
        coordinates.push_back(at::tensor(evidence[name]));
        transform(parents.begin(), parents.end(), back_inserter(coordinates), [&evidence](const auto& parent) { return at::tensor(evidence[parent->getName()]); });
        return cpTable.index({ coordinates }).item<float>();
    }
    vector<string> Node::graph(const string& className)
    {
        auto output = vector<string>();
        auto suffix = name == className ? ", fontcolor=red, fillcolor=lightblue, style=filled " : "";
        output.push_back(name + " [shape=circle" + suffix + "] \n");
        transform(children.begin(), children.end(), back_inserter(output), [this](const auto& child) { return name + " -> " + child->getName(); });
        return output;
    }
}