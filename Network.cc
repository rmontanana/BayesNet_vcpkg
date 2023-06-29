#include "Network.h"
namespace bayesnet {
    Network::~Network()
    {
        for (auto& pair : nodes) {
            delete pair.second;
        }
    }
    void Network::addNode(std::string name, int numStates)
    {
        nodes[name] = new Node(name, numStates);
    }
    void Network::addEdge(const std::string parent, const std::string child)
    {
        if (nodes.find(parent) == nodes.end()) {
            throw std::invalid_argument("Parent node " + parent + " does not exist");
        }
        if (nodes.find(child) == nodes.end()) {
            throw std::invalid_argument("Child node " + child + " does not exist");
        }
        nodes[parent]->addChild(nodes[child]);
        nodes[child]->addParent(nodes[parent]);
    }
    std::map<std::string, Node*>& Network::getNodes()
    {
        return nodes;
    }
    void Network::fit(const std::vector<std::vector<int>>& dataset, const int smoothing)
    {
        auto jointCounts = [](const std::vector<std::vector<int>>& data, const std::vector<int>& indices, int numStates) {
            int size = indices.size();
            std::vector<int64_t> sizes(size, numStates);
            torch::Tensor counts = torch::zeros(sizes, torch::kLong);

            for (const auto& row : data) {
                int idx = 0;
                for (int i = 0; i < size; ++i) {
                    idx = idx * numStates + row[indices[i]];
                }
                counts.view({ -1 }).add_(idx, 1);
            }

            return counts;
        };

        auto marginalCounts = [](const torch::Tensor& jointCounts) {
            return jointCounts.sum(-1);
        };

        for (auto& pair : nodes) {
            Node* node = pair.second;

            std::vector<int> indices;
            for (const auto& parent : node->getParents()) {
                indices.push_back(nodes[parent->getName()]->getId());
            }
            indices.push_back(node->getId());

            for (auto& child : node->getChildren()) {
                torch::Tensor counts = jointCounts(dataset, indices, node->getNumStates()) + smoothing;
                torch::Tensor parentCounts = marginalCounts(counts);
                parentCounts = parentCounts.unsqueeze(-1);

                torch::Tensor cpt = counts.to(torch::kDouble) / parentCounts.to(torch::kDouble);
                setCPD(node->getCPDKey(child), cpt);
            }
        }
    }

    torch::Tensor& Network::getCPD(const std::string& key)
    {
        return cpds[key];
    }

    void Network::setCPD(const std::string& key, const torch::Tensor& cpt)
    {
        cpds[key] = cpt;
    }
}
