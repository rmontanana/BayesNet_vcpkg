#include "Network.h"
namespace bayesnet {
    Network::Network() : laplaceSmoothing(1), root(nullptr), features(vector<string>()), className("") {}
    Network::Network(int smoothing) : laplaceSmoothing(smoothing), root(nullptr), features(vector<string>()), className("") {}
    Network::~Network()
    {
        for (auto& pair : nodes) {
            delete pair.second;
        }
    }
    void Network::addNode(string name, int numStates)
    {
        if (nodes.find(name) != nodes.end()) {
            throw invalid_argument("Node " + name + " already exists");
        }
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
            // remove problematic edge
            nodes[parent]->removeChild(nodes[child]);
            nodes[child]->removeParent(nodes[parent]);
            throw invalid_argument("Adding this edge forms a cycle in the graph.");
        }

    }
    map<string, Node*>& Network::getNodes()
    {
        return nodes;
    }
    void Network::buildNetwork()
    {
        // Add features as nodes to the network
        for (int i = 0; i < features.size(); ++i) {
            addNode(features[i], *max_element(dataset[features[i]].begin(), dataset[features[i]].end()) + 1);
        }
        // Add class as node to the network
        addNode(className, *max_element(dataset[className].begin(), dataset[className].end()) + 1);
        // Add edges from class to features => naive Bayes
        for (auto feature : features) {
            addEdge(className, feature);
        }
        addEdge("petalwidth", "petallength");
    }
    void Network::fit(const vector<vector<int>>& dataset, const vector<int>& labels, const vector<string>& featureNames, const string& className)
    {
        features = featureNames;
        this->className = className;
        // Build dataset
        for (int i = 0; i < featureNames.size(); ++i) {
            this->dataset[featureNames[i]] = dataset[i];
        }
        this->dataset[className] = labels;
        buildNetwork();
        estimateParameters();
    }

    // void Network::estimateParameters()
    // {
    //     auto dimensions = vector<int64_t>();
    //     for (auto [name, node] : nodes) {
    //         // Get dimensions of the CPT
    //         dimensions.clear();
    //         dimensions.push_back(node->getNumStates());
    //         for (auto father : node->getParents()) {
    //             dimensions.push_back(father->getNumStates());
    //         }
    //         auto length = dimensions.size();
    //         // Create a tensor of zeros with the dimensions of the CPT
    //         torch::Tensor cpt = torch::zeros(dimensions, torch::kFloat);
    //         // Fill table with counts
    //         for (int n_sample = 0; n_sample < dataset[name].size(); ++n_sample) {
    //             torch::List<c10::optional<torch::Tensor>> coordinates;
    //             coordinates.push_back(torch::tensor(dataset[name][n_sample]));
    //             for (auto father : node->getParents()) {
    //                 coordinates.push_back(torch::tensor(dataset[father->getName()][n_sample]));
    //             }
    //             // Increment the count of the corresponding coordinate
    //             cpt.index_put_({ coordinates }, cpt.index({ coordinates }) + 1);
    //         }
    //         // store thre resulting cpt in the node
    //         node->setCPT(cpt);
    //     }
    // }

    // void Network::estimateParameters()
    // {
    //     // Lambda function to compute joint counts of states
    //     auto jointCounts = [this](const vector<string>& nodeNames) {
    //         int size = nodeNames.size();
    //         std::vector<int64_t> sizes(size);

    //         for (int i = 0; i < size; ++i) {
    //             sizes[i] = this->nodes[nodeNames[i]]->getNumStates();
    //         }

    //         torch::Tensor counts = torch::zeros(sizes, torch::kLong);

    //         int dataSize = this->dataset[nodeNames[0]].size();

    //         for (int dataIdx = 0; dataIdx < dataSize; ++dataIdx) {
    //             std::vector<torch::Tensor> idx(size);
    //             for (int i = 0; i < size; ++i) {
    //                 idx[i] = torch::tensor(this->dataset[nodeNames[i]][dataIdx], torch::kLong);
    //             }
    //             torch::Tensor indices = torch::stack(idx);
    //             counts.index_put_({ indices }, counts.index({ indices }) + 1);
    //         }

    //         return counts;
    //         };

    //     // Lambda function to compute marginal counts of states
    //     auto marginalCounts = [](const torch::Tensor& jointCounts) {
    //         return jointCounts.sum(-1);
    //         };

    //     for (auto& pair : nodes) {
    //         Node* node = pair.second;

    //         // Create a list of names of the node and its parents
    //         std::vector<string> nodeNames;
    //         nodeNames.push_back(node->getName());
    //         for (Node* parent : node->getParents()) {
    //             nodeNames.push_back(parent->getName());
    //         }

    //         // Compute counts and normalize to get probabilities
    //         torch::Tensor counts = jointCounts(nodeNames) + laplaceSmoothing;
    //         torch::Tensor parentCounts = marginalCounts(counts);
    //         parentCounts = parentCounts.unsqueeze(-1);

    //         // The CPT is represented as a tensor and stored in the Node
    //         node->setCPT((counts.to(torch::kDouble) / parentCounts.to(torch::kDouble)));
    //     }
    // }
    void Network::estimateParameters()
    {
        // Lambda function to compute joint counts of states
        auto jointCounts = [this](const vector<string>& nodeNames) {
            int size = nodeNames.size();
            std::vector<int64_t> sizes(size);

            for (int i = 0; i < size; ++i) {
                sizes[i] = this->nodes[nodeNames[i]]->getNumStates();
            }

            torch::Tensor counts = torch::zeros(sizes, torch::kLong);

            int dataSize = this->dataset[nodeNames[0]].size();
            torch::List<c10::optional<torch::Tensor>> indices;
            for (int dataIdx = 0; dataIdx < dataSize; ++dataIdx) {
                indices.clear();
                for (int i = 0; i < size; ++i) {
                    indices.push_back(torch::tensor(this->dataset[nodeNames[i]][dataIdx], torch::kLong));
                }
                //torch::Tensor indices = torch::stack(idx);
                counts.index_put_({ indices }, counts.index({ indices }) + 1);
            }

            return counts;
            };

        // Lambda function to compute marginal counts of states
        auto marginalCounts = [](const torch::Tensor& jointCounts) {
            return jointCounts.sum(-1);
            };

        for (auto& pair : nodes) {
            Node* node = pair.second;

            // Create a list of names of the node and its parents
            std::vector<string> nodeNames;
            nodeNames.push_back(node->getName());
            for (Node* parent : node->getParents()) {
                nodeNames.push_back(parent->getName());
            }

            // Compute counts and normalize to get probabilities
            torch::Tensor counts = jointCounts(nodeNames) + laplaceSmoothing;
            torch::Tensor parentCounts = marginalCounts(counts);
            parentCounts = parentCounts.unsqueeze(-1);

            // The CPT is represented as a tensor and stored in the Node
            node->setCPT((counts.to(torch::kDouble) / parentCounts.to(torch::kDouble)));
        }
    }

}
