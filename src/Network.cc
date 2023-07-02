#include "Network.h"
namespace bayesnet {
    Network::Network() : laplaceSmoothing(1), root(nullptr), features(vector<string>()), className("") {}
    Network::Network(int smoothing) : laplaceSmoothing(smoothing), root(nullptr), features(vector<string>()), className("") {}
    Network::Network(Network& other) : laplaceSmoothing(other.laplaceSmoothing), root(other.root), features(other.features), className(other.className)
    {
        for (auto& pair : other.nodes) {
            nodes[pair.first] = new Node(*pair.second);
        }
    }
    Network::~Network()
    {
        for (auto& pair : nodes) {
            delete pair.second;
        }
    }
    void Network::addNode(string name, int numStates)
    {
        if (nodes.find(name) != nodes.end()) {
            // if node exists update its number of states
            nodes[name]->setNumStates(numStates);
            return;
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
    void Network::fit(const vector<vector<int>>& dataset, const vector<int>& labels, const vector<string>& featureNames, const string& className)
    {
        features = featureNames;
        this->className = className;
        // Build dataset
        for (int i = 0; i < featureNames.size(); ++i) {
            this->dataset[featureNames[i]] = dataset[i];
        }
        this->dataset[className] = labels;
        estimateParameters();
    }

    void Network::estimateParameters()
    {
        auto dimensions = vector<int64_t>();
        for (auto [name, node] : nodes) {
            // Get dimensions of the CPT
            dimensions.clear();
            dimensions.push_back(node->getNumStates());
            for (auto father : node->getParents()) {
                dimensions.push_back(father->getNumStates());
            }
            auto length = dimensions.size();
            // Create a tensor of zeros with the dimensions of the CPT
            torch::Tensor cpt = torch::zeros(dimensions, torch::kFloat) + laplaceSmoothing;
            // Fill table with counts
            for (int n_sample = 0; n_sample < dataset[name].size(); ++n_sample) {
                torch::List<c10::optional<torch::Tensor>> coordinates;
                coordinates.push_back(torch::tensor(dataset[name][n_sample]));
                for (auto father : node->getParents()) {
                    coordinates.push_back(torch::tensor(dataset[father->getName()][n_sample]));
                }
                // Increment the count of the corresponding coordinate
                cpt.index_put_({ coordinates }, cpt.index({ coordinates }) + 1);
            }
            // Normalize the counts
            cpt = cpt / cpt.sum(0);
            // store thre resulting cpt in the node
            node->setCPT(cpt);
        }
    }
    // pair<int, double> Network::predict_sample(const vector<int>& sample)
    // {


    //     // For each possible class, calculate the posterior probability
    //     Node* classNode = nodes[className];
    //     int numClassStates = classNode->getNumStates();
    //     vector<double> classProbabilities(numClassStates, 0.0);
    //     for (int classState = 0; classState < numClassStates; ++classState) {
    //         // Start with the prior probability of the class
    //         classProbabilities[classState] = classNode->getCPT()[classState].item<double>();

    //         // Multiply by the likelihood of each feature given the class
    //         for (auto& pair : nodes) {
    //             if (pair.first != className) {
    //                 Node* node = pair.second;
    //                 int featureValue = featureValues[pair.first];

    //                 // We use the class as the parent state to index into the CPT
    //                 classProbabilities[classState] *= node->getCPT()[classState][featureValue].item<double>();
    //             }
    //         }
    //     }

    //     // Find the class with the maximum posterior probability
    //     auto maxElem = max_element(classProbabilities.begin(), classProbabilities.end());
    //     int predictedClass = distance(classProbabilities.begin(), maxElem);
    //     double maxProbability = *maxElem;

    //     return make_pair(predictedClass, maxProbability);
    // }
    vector<int> Network::predict(const vector<vector<int>>& samples)
    {
        vector<int> predictions;
        vector<int> sample;
        for (int row = 0; row < samples[0].size(); ++row) {
            sample.clear();
            for (int col = 0; col < samples.size(); ++col) {
                sample.push_back(samples[col][row]);
            }
            predictions.push_back(predict_sample(sample).first);
        }
        return predictions;
    }
    vector<pair<int, double>> Network::predict_proba(const vector<vector<int>>& samples)
    {
        vector<pair<int, double>> predictions;
        vector<int> sample;
        for (int row = 0; row < samples[0].size(); ++row) {
            sample.clear();
            for (int col = 0; col < samples.size(); ++col) {
                sample.push_back(samples[col][row]);
            }
            predictions.push_back(predict_sample(sample));
        }
        return predictions;
    }
    double Network::score(const vector<vector<int>>& samples, const vector<int>& labels)
    {
        vector<int> y_pred = predict(samples);
        int correct = 0;
        for (int i = 0; i < y_pred.size(); ++i) {
            if (y_pred[i] == labels[i]) {
                correct++;
            }
        }
        return (double)correct / y_pred.size();
    }
    pair<int, double> Network::predict_sample(const vector<int>& sample)
    {
        // Ensure the sample size is equal to the number of features
        if (sample.size() != features.size()) {
            throw invalid_argument("Sample size (" + to_string(sample.size()) +
                ") does not match the number of features (" + to_string(features.size()) + ")");
        }
        // Map the feature values to their corresponding nodes
        map<string, int> featureValues;
        for (int i = 0; i < features.size(); ++i) {
            featureValues[features[i]] = sample[i];
        }

        // For each possible class, calculate the posterior probability
        Network network = *this;
        vector<double> classProbabilities = eliminateVariables(network, featureValues);

        // Normalize the probabilities to sum to 1
        double sum = accumulate(classProbabilities.begin(), classProbabilities.end(), 0.0);
        for (double& prob : classProbabilities) {
            prob /= sum;
        }
        // Find the class with the maximum posterior probability
        auto maxElem = max_element(classProbabilities.begin(), classProbabilities.end());
        int predictedClass = distance(classProbabilities.begin(), maxElem);
        double maxProbability = *maxElem;

        return make_pair(predictedClass, maxProbability);
    }
    vector<double> eliminateVariables(network, featureValues)
    {
        
    }
}
