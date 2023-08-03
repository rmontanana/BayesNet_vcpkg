#include <thread>
#include <mutex>
#include "Network.h"
#include "bayesnetUtils.h"
namespace bayesnet {
    Network::Network() : features(vector<string>()), className(""), classNumStates(0), fitted(false) {}
    Network::Network(float maxT) : features(vector<string>()), className(""), classNumStates(0), maxThreads(maxT), fitted(false) {}
    Network::Network(float maxT, int smoothing) : laplaceSmoothing(smoothing), features(vector<string>()), className(""), classNumStates(0), maxThreads(maxT), fitted(false) {}
    Network::Network(Network& other) : laplaceSmoothing(other.laplaceSmoothing), features(other.features), className(other.className), classNumStates(other.getClassNumStates()), maxThreads(other.
        getmaxThreads()), fitted(other.fitted)
    {
        for (const auto& pair : other.nodes) {
            nodes[pair.first] = std::make_unique<Node>(*pair.second);
        }
    }
    void Network::initialize()
    {
        features = vector<string>();
        className = "";
        classNumStates = 0;
        fitted = false;
        nodes.clear();
        dataset.clear();
        samples = torch::Tensor();
    }
    float Network::getmaxThreads()
    {
        return maxThreads;
    }
    torch::Tensor& Network::getSamples()
    {
        return samples;
    }
    void Network::addNode(const string& name, int numStates)
    {
        if (name == "") {
            throw invalid_argument("Node name cannot be empty");
        }
        if (find(features.begin(), features.end(), name) == features.end()) {
            features.push_back(name);
        }
        if (nodes.find(name) != nodes.end()) {
            // if node exists update its number of states and remove parents, children and CPT
            nodes[name]->clear();
            nodes[name]->setNumStates(numStates);
            return;
        }
        nodes[name] = std::make_unique<Node>(name, numStates);
    }
    vector<string> Network::getFeatures()
    {
        return features;
    }
    int Network::getClassNumStates()
    {
        return classNumStates;
    }
    int Network::getStates()
    {
        int result = 0;
        for (auto& node : nodes) {
            result += node.second->getNumStates();
        }
        return result;
    }
    string Network::getClassName()
    {
        return className;
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
    void Network::addEdge(const string& parent, const string& child)
    {
        if (nodes.find(parent) == nodes.end()) {
            throw invalid_argument("Parent node " + parent + " does not exist");
        }
        if (nodes.find(child) == nodes.end()) {
            throw invalid_argument("Child node " + child + " does not exist");
        }
        // Temporarily add edge to check for cycles
        nodes[parent]->addChild(nodes[child].get());
        nodes[child]->addParent(nodes[parent].get());
        unordered_set<string> visited;
        unordered_set<string> recStack;
        if (isCyclic(nodes[child]->getName(), visited, recStack)) // if adding this edge forms a cycle
        {
            // remove problematic edge
            nodes[parent]->removeChild(nodes[child].get());
            nodes[child]->removeParent(nodes[parent].get());
            throw invalid_argument("Adding this edge forms a cycle in the graph.");
        }
    }
    map<string, std::unique_ptr<Node>>& Network::getNodes()
    {
        return nodes;
    }
    void Network::checkFitData(int n_samples, int n_features, int n_samples_y, const vector<string>& featureNames, const string& className)
    {
        if (n_samples != n_samples_y) {
            throw invalid_argument("X and y must have the same number of samples in Network::fit (" + to_string(n_samples) + " != " + to_string(n_samples_y) + ")");
        }
        if (n_features != featureNames.size()) {
            throw invalid_argument("X and features must have the same number of features in Network::fit (" + to_string(n_features) + " != " + to_string(featureNames.size()) + ")");
        }
        if (n_features != features.size() - 1) {
            throw invalid_argument("X and local features must have the same number of features in Network::fit (" + to_string(n_features) + " != " + to_string(features.size() - 1) + ")");
        }
        if (find(features.begin(), features.end(), className) == features.end()) {
            throw invalid_argument("className not found in Network::features");
        }
        for (auto& feature : featureNames) {
            if (find(features.begin(), features.end(), feature) == features.end()) {
                throw invalid_argument("Feature " + feature + " not found in Network::features");
            }
        }
    }
    // X comes in nxm, where n is the number of features and m the number of samples
    void Network::fit(torch::Tensor& X, torch::Tensor& y, const vector<string>& featureNames, const string& className)
    {
        checkFitData(X.size(1), X.size(0), y.size(0), featureNames, className);
        this->className = className;
        dataset.clear();
        // Specific part
        classNumStates = torch::max(y).item<int>() + 1;
        Tensor ytmp = torch::transpose(y.view({ y.size(0), 1 }), 0, 1);
        samples = torch::cat({ X , ytmp }, 0);
        for (int i = 0; i < featureNames.size(); ++i) {
            auto row_feature = X.index({ i, "..." });
            dataset[featureNames[i]] = vector<int>(row_feature.data_ptr<int>(), row_feature.data_ptr<int>() + row_feature.size(0));;
        }
        dataset[className] = vector<int>(y.data_ptr<int>(), y.data_ptr<int>() + y.size(0));
        completeFit();
    }
    // input_data comes in nxm, where n is the number of features and m the number of samples
    void Network::fit(const vector<vector<int>>& input_data, const vector<int>& labels, const vector<string>& featureNames, const string& className)
    {
        checkFitData(input_data[0].size(), input_data.size(), labels.size(), featureNames, className);
        this->className = className;
        dataset.clear();
        // Specific part
        classNumStates = *max_element(labels.begin(), labels.end()) + 1;
        // Build dataset & tensor of samples (nxm) (n+1 because of the class)
        samples = torch::zeros({ static_cast<int>(input_data.size() + 1), static_cast<int>(input_data[0].size()) }, torch::kInt32);
        for (int i = 0; i < featureNames.size(); ++i) {
            dataset[featureNames[i]] = input_data[i];
            samples.index_put_({ i, "..." }, torch::tensor(input_data[i], torch::kInt32));
        }
        dataset[className] = labels;
        samples.index_put_({ -1, "..." }, torch::tensor(labels, torch::kInt32));
        completeFit();
    }
    void Network::completeFit()
    {
        int maxThreadsRunning = static_cast<int>(std::thread::hardware_concurrency() * maxThreads);
        if (maxThreadsRunning < 1) {
            maxThreadsRunning = 1;
        }
        vector<thread> threads;
        mutex mtx;
        condition_variable cv;
        int activeThreads = 0;
        int nextNodeIndex = 0;
        while (nextNodeIndex < nodes.size()) {
            unique_lock<mutex> lock(mtx);
            cv.wait(lock, [&activeThreads, &maxThreadsRunning]() { return activeThreads < maxThreadsRunning; });
            threads.emplace_back([this, &nextNodeIndex, &mtx, &cv, &activeThreads]() {
                while (true) {
                    unique_lock<mutex> lock(mtx);
                    if (nextNodeIndex >= nodes.size()) {
                        break;  // No more work remaining
                    }
                    auto& pair = *std::next(nodes.begin(), nextNodeIndex);
                    ++nextNodeIndex;
                    lock.unlock();
                    pair.second->computeCPT(dataset, laplaceSmoothing);
                    lock.lock();
                    nodes[pair.first] = std::move(pair.second);
                    lock.unlock();
                }
                lock_guard<mutex> lock(mtx);
                --activeThreads;
                cv.notify_one();
                });
            ++activeThreads;
        }
        for (auto& thread : threads) {
            thread.join();
        }
        fitted = true;
    }
    torch::Tensor Network::predict_tensor(const torch::Tensor& samples, const bool proba)
    {
        if (!fitted) {
            throw logic_error("You must call fit() before calling predict()");
        }
        torch::Tensor result;
        result = torch::zeros({ samples.size(1), classNumStates }, torch::kFloat64);
        for (int i = 0; i < samples.size(1); ++i) {
            auto sample = samples.index({ "...", i });
            result.index_put_({ i, "..." }, torch::tensor(predict_sample(sample), torch::kFloat64));
        }
        if (proba)
            return result;
        else
            return result.argmax(1);
    }
    // Return mxn tensor of probabilities
    Tensor Network::predict_proba(const Tensor& samples)
    {
        return predict_tensor(samples, true);
    }

    // Return mxn tensor of probabilities
    Tensor Network::predict(const Tensor& samples)
    {
        return predict_tensor(samples, false);
    }

    // Return mx1 vector of predictions
    // tsamples is nxm vector of samples
    vector<int> Network::predict(const vector<vector<int>>& tsamples)
    {
        if (!fitted) {
            throw logic_error("You must call fit() before calling predict()");
        }
        vector<int> predictions;
        vector<int> sample;
        for (int row = 0; row < tsamples[0].size(); ++row) {
            sample.clear();
            for (int col = 0; col < tsamples.size(); ++col) {
                sample.push_back(tsamples[col][row]);
            }
            vector<double> classProbabilities = predict_sample(sample);
            // Find the class with the maximum posterior probability
            auto maxElem = max_element(classProbabilities.begin(), classProbabilities.end());
            int predictedClass = distance(classProbabilities.begin(), maxElem);
            predictions.push_back(predictedClass);
        }
        return predictions;
    }
    // Return mxn vector of probabilities
    vector<vector<double>> Network::predict_proba(const vector<vector<int>>& tsamples)
    {
        if (!fitted) {
            throw logic_error("You must call fit() before calling predict_proba()");
        }
        vector<vector<double>> predictions;
        vector<int> sample;
        for (int row = 0; row < tsamples[0].size(); ++row) {
            sample.clear();
            for (int col = 0; col < tsamples.size(); ++col) {
                sample.push_back(tsamples[col][row]);
            }
            predictions.push_back(predict_sample(sample));
        }
        return predictions;
    }
    double Network::score(const vector<vector<int>>& tsamples, const vector<int>& labels)
    {
        vector<int> y_pred = predict(tsamples);
        int correct = 0;
        for (int i = 0; i < y_pred.size(); ++i) {
            if (y_pred[i] == labels[i]) {
                correct++;
            }
        }
        return (double)correct / y_pred.size();
    }
    // Return 1xn vector of probabilities
    vector<double> Network::predict_sample(const vector<int>& sample)
    {
        // Ensure the sample size is equal to the number of features
        if (sample.size() != features.size() - 1) {
            throw invalid_argument("Sample size (" + to_string(sample.size()) +
                ") does not match the number of features (" + to_string(features.size() - 1) + ")");
        }
        map<string, int> evidence;
        for (int i = 0; i < sample.size(); ++i) {
            evidence[features[i]] = sample[i];
        }
        return exactInference(evidence);
    }
    // Return 1xn vector of probabilities
    vector<double> Network::predict_sample(const Tensor& sample)
    {
        // Ensure the sample size is equal to the number of features
        if (sample.size(0) != features.size() - 1) {
            throw invalid_argument("Sample size (" + to_string(sample.size(0)) +
                ") does not match the number of features (" + to_string(features.size() - 1) + ")");
        }
        map<string, int> evidence;
        for (int i = 0; i < sample.size(0); ++i) {
            evidence[features[i]] = sample[i].item<int>();
        }
        return exactInference(evidence);
    }
    double Network::computeFactor(map<string, int>& completeEvidence)
    {
        double result = 1.0;
        for (auto& node : getNodes()) {
            result *= node.second->getFactorValue(completeEvidence);
        }
        return result;
    }
    vector<double> Network::exactInference(map<string, int>& evidence)
    {
        vector<double> result(classNumStates, 0.0);
        vector<thread> threads;
        mutex mtx;
        for (int i = 0; i < classNumStates; ++i) {
            threads.emplace_back([this, &result, &evidence, i, &mtx]() {
                auto completeEvidence = map<string, int>(evidence);
                completeEvidence[getClassName()] = i;
                double factor = computeFactor(completeEvidence);
                lock_guard<mutex> lock(mtx);
                result[i] = factor;
                });
        }
        for (auto& thread : threads) {
            thread.join();
        }

        // Normalize result
        double sum = accumulate(result.begin(), result.end(), 0.0);
        transform(result.begin(), result.end(), result.begin(), [sum](double& value) { return value / sum; });
        return result;
    }
    vector<string> Network::show()
    {
        vector<string> result;
        // Draw the network
        for (auto& node : nodes) {
            string line = node.first + " -> ";
            for (auto child : node.second->getChildren()) {
                line += child->getName() + ", ";
            }
            result.push_back(line);
        }
        return result;
    }
    vector<string> Network::graph(const string& title)
    {
        auto output = vector<string>();
        auto prefix = "digraph BayesNet {\nlabel=<BayesNet ";
        auto suffix = ">\nfontsize=30\nfontcolor=blue\nlabelloc=t\nlayout=circo\n";
        string header = prefix + title + suffix;
        output.push_back(header);
        for (auto& node : nodes) {
            auto result = node.second->graph(className);
            output.insert(output.end(), result.begin(), result.end());
        }
        output.push_back("}\n");
        return output;
    }
    vector<pair<string, string>> Network::getEdges()
    {
        auto edges = vector<pair<string, string>>();
        for (const auto& node : nodes) {
            auto head = node.first;
            for (const auto& child : node.second->getChildren()) {
                auto tail = child->getName();
                edges.push_back({ head, tail });
            }
        }
        return edges;
    }
    vector<string> Network::topological_sort()
    {
        /* Check if al the fathers of every node are before the node */
        auto result = features;
        result.erase(remove(result.begin(), result.end(), className), result.end());
        bool ending{ false };
        int idx = 0;
        while (!ending) {
            ending = true;
            for (auto feature : features) {
                auto fathers = nodes[feature]->getParents();
                for (const auto& father : fathers) {
                    auto fatherName = father->getName();
                    if (fatherName == className) {
                        continue;
                    }
                    // Check if father is placed before the actual feature
                    auto it = find(result.begin(), result.end(), fatherName);
                    if (it != result.end()) {
                        auto it2 = find(result.begin(), result.end(), feature);
                        if (it2 != result.end()) {
                            if (distance(it, it2) < 0) {
                                // if it is not, insert it before the feature
                                result.erase(remove(result.begin(), result.end(), fatherName), result.end());
                                result.insert(it2, fatherName);
                                ending = false;
                            }
                        } else {
                            throw logic_error("Error in topological sort because of node " + feature + " is not in result");
                        }
                    } else {
                        throw logic_error("Error in topological sort because of node father " + fatherName + " is not in result");
                    }
                }
            }
        }
        return result;
    }
    void Network::dump_cpt()
    {
        for (auto& node : nodes) {
            cout << "* " << node.first << ": " << node.second->getCPT() << endl;
        }
    }
}
