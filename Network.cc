#include "Network.h"
namespace bayesnet {
    Network::Network() : laplaceSmoothing(1), root(nullptr) {}
    Network::Network(int smoothing) : laplaceSmoothing(smoothing), root(nullptr) {}
    Network::~Network()
    {
        for (auto& pair : nodes) {
            delete pair.second;
        }
    }
    void Network::addNode(string name, int numStates)
    {
        nodes[name] = new Node(name, numStates);
        if (root == nullptr) {
            root = nodes[name];
        }
    }
    void Network::setRoot(string name)
    {
        if (nodes.find(name) == nodes.end()) {
            throw invalid_argument("Node " + name + " does not exist");
        }
        root = nodes[name];
    }
    Node* Network::getRoot()
    {
        return root;
    }
    bool Network::isCyclic(const string& nodeId, unordered_set<string>& visited, unordered_set<string>& recStack)
    {
        if (visited.find(nodeId) == visited.end()) // if node hasn't been visited yet
        {
            visited.insert(nodeId);
            recStack.insert(nodeId);

            for (Node* child : nodes[nodeId]->getChildren()) {
                if (visited.find(child->getName()) == visited.end() && isCyclic(child->getName(), visited, recStack))
                    return true;
                else if (recStack.find(child->getName()) != recStack.end())
                    return true;
            }
        }
        recStack.erase(nodeId); // remove node from recursion stack before function ends
        return false;
    }
    void Network::addEdge(const string parent, const string child)
    {
        if (nodes.find(parent) == nodes.end()) {
            throw invalid_argument("Parent node " + parent + " does not exist");
        }
        if (nodes.find(child) == nodes.end()) {
            throw invalid_argument("Child node " + child + " does not exist");
        }
        // Temporarily add edge to check for cycles
        nodes[parent]->addChild(nodes[child]);
        nodes[child]->addParent(nodes[parent]);
        // temporarily add edge

        unordered_set<string> visited;
        unordered_set<string> recStack;

        if (isCyclic(nodes[child]->getName(), visited, recStack)) // if adding this edge forms a cycle
        {
            // remove edge
            nodes[parent]->removeChild(nodes[child]);
            nodes[child]->removeParent(nodes[parent]);
            throw invalid_argument("Adding this edge forms a cycle in the graph.");
        }

    }
    map<string, Node*>& Network::getNodes()
    {
        return nodes;
    }
    void Network::buildNetwork(const vector<vector<int>>& dataset, const vector<int>& labels, const vector<string>& featureNames, const string& className)
    {
        // Add features as nodes to the network
        for (int i = 0; i < featureNames.size(); ++i) {
            addNode(featureNames[i], *max_element(dataset[i].begin(), dataset[i].end()) + 1);
        }
        // Add class as node to the network
        addNode(className, *max_element(labels.begin(), labels.end()) + 1);
        // Add edges from class to features => naive Bayes
        for (auto feature : featureNames) {
            addEdge(className, feature);
        }
    }
    void Network::fit(const vector<vector<int>>& dataset, const vector<int>& labels, const vector<string>& featureNames, const string& className)
    {
        buildNetwork(dataset, labels, featureNames, className);
        //estimateParameters(dataset);

        // auto jointCounts = [](const vector<vector<int>>& data, const vector<int>& indices, int numStates) {
        //     int size = indices.size();
        //     vector<int64_t> sizes(size, numStates);
        //     torch::Tensor counts = torch::zeros(sizes, torch::kLong);

        //     for (const auto& row : data) {
        //         int idx = 0;
        //         for (int i = 0; i < size; ++i) {
        //             idx = idx * numStates + row[indices[i]];
        //         }
        //         counts.view({ -1 }).add_(idx, 1);
        //     }

        //     return counts;
        //     };

        // auto marginalCounts = [](const torch::Tensor& jointCounts) {
        //     return jointCounts.sum(-1);
        //     };

        // for (auto& pair : nodes) {
        //     Node* node = pair.second;

        //     vector<int> indices;
        //     for (const auto& parent : node->getParents()) {
        //         indices.push_back(nodes[parent->getName()]->getId());
        //     }
        //     indices.push_back(node->getId());

        //     for (auto& child : node->getChildren()) {
        //         torch::Tensor counts = jointCounts(dataset, indices, node->getNumStates()) + laplaceSmoothing;
        //         torch::Tensor parentCounts = marginalCounts(counts);
        //         parentCounts = parentCounts.unsqueeze(-1);

        //         torch::Tensor cpt = counts.to(torch::kDouble) / parentCounts.to(torch::kDouble);
        //         setCPD(node->getCPDKey(child), cpt);
        //     }
        // }
    }

    torch::Tensor& Network::getCPD(const string& key)
    {
        return cpds[key];
    }

    void Network::setCPD(const string& key, const torch::Tensor& cpt)
    {
        cpds[key] = cpt;
    }
}
