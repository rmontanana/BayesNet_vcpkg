#ifndef NETWORK_H
#define NETWORK_H
#include "Node.h"
#include <map>
#include <vector>

namespace bayesnet {
    class Network {
    private:
        map<string, unique_ptr<Node>> nodes;
        bool fitted;
        float maxThreads = 0.95;
        int classNumStates;
        vector<string> features; // Including classname
        string className;
        double laplaceSmoothing;
        torch::Tensor samples; // nxm tensor used to fit the model
        bool isCyclic(const std::string&, std::unordered_set<std::string>&, std::unordered_set<std::string>&);
        vector<double> predict_sample(const vector<int>&);
        vector<double> predict_sample(const torch::Tensor&);
        vector<double> exactInference(map<string, int>&);
        double computeFactor(map<string, int>&);
        void completeFit(const map<string, vector<int>>& states, const torch::Tensor& weights);
        void checkFitData(int n_features, int n_samples, int n_samples_y, const vector<string>& featureNames, const string& className, const map<string, vector<int>>& states, const torch::Tensor& weights);
        void setStates(const map<string, vector<int>>&);
    public:
        Network();
        explicit Network(float);
        explicit Network(Network&);
        ~Network() = default;
        torch::Tensor& getSamples();
        float getmaxThreads();
        void addNode(const string&);
        void addEdge(const string&, const string&);
        map<string, std::unique_ptr<Node>>& getNodes();
        vector<string> getFeatures() const;
        int getStates() const;
        vector<pair<string, string>> getEdges() const;
        int getNumEdges() const;
        int getClassNumStates() const;
        string getClassName() const;
        /*
        Notice: Nodes have to be inserted in the same order as they are in the dataset, i.e., first node is first column and so on.
        */
        void fit(const vector<vector<int>>& input_data, const vector<int>& labels, const vector<double>& weights, const vector<string>& featureNames, const string& className, const map<string, vector<int>>& states);
        void fit(const torch::Tensor& X, const torch::Tensor& y, const torch::Tensor& weights, const vector<string>& featureNames, const string& className, const map<string, vector<int>>& states);
        void fit(const torch::Tensor& samples, const torch::Tensor& weights, const vector<string>& featureNames, const string& className, const map<string, vector<int>>& states);
        vector<int> predict(const vector<vector<int>>&); // Return mx1 vector of predictions
        torch::Tensor predict(const torch::Tensor&); // Return mx1 tensor of predictions
        torch::Tensor predict_tensor(const torch::Tensor& samples, const bool proba);
        vector<vector<double>> predict_proba(const vector<vector<int>>&); // Return mxn vector of probabilities
        torch::Tensor predict_proba(const torch::Tensor&); // Return mxn tensor of probabilities
        double score(const vector<vector<int>>&, const vector<int>&);
        vector<string> topological_sort();
        vector<string> show() const;
        vector<string> graph(const string& title) const; // Returns a vector of strings representing the graph in graphviz format
        void initialize();
        void dump_cpt() const;
        inline string version() { return "0.2.0"; }
    };
}
#endif