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
        int laplaceSmoothing = 1;
        torch::Tensor samples; // nxm tensor used to fit the model
        bool isCyclic(const std::string&, std::unordered_set<std::string>&, std::unordered_set<std::string>&);
        vector<double> predict_sample(const vector<int>&);
        vector<double> predict_sample(const torch::Tensor&);
        vector<double> exactInference(map<string, int>&);
        double computeFactor(map<string, int>&);
        double mutual_info(torch::Tensor&, torch::Tensor&);
        double entropy(torch::Tensor&);
        double conditionalEntropy(torch::Tensor&, torch::Tensor&);
        double mutualInformation(torch::Tensor&, torch::Tensor&);
        void completeFit();
        void checkFitData(int n_features, int n_samples, int n_samples_y, const vector<string>& featureNames, const string& className);
        void setStates();
    public:
        Network();
        explicit Network(float, int);
        explicit Network(float);
        explicit Network(Network&);
        torch::Tensor& getSamples();
        float getmaxThreads();
        void addNode(const string&);
        void addEdge(const string&, const string&);
        map<string, std::unique_ptr<Node>>& getNodes();
        vector<string> getFeatures();
        int getStates();
        vector<pair<string, string>> getEdges();
        int getClassNumStates();
        string getClassName();
        void fit(const vector<vector<int>>&, const vector<int>&, const vector<string>&, const string&);
        void fit(const torch::Tensor&, const torch::Tensor&, const vector<string>&, const string&);
        void fit(const torch::Tensor&, const vector<string>&, const string&);
        vector<int> predict(const vector<vector<int>>&); // Return mx1 vector of predictions
        torch::Tensor predict(const torch::Tensor&); // Return mx1 tensor of predictions
        //Computes the conditional edge weight of variable index u and v conditioned on class_node
        torch::Tensor conditionalEdgeWeight();
        torch::Tensor predict_tensor(const torch::Tensor& samples, const bool proba);
        vector<vector<double>> predict_proba(const vector<vector<int>>&); // Return mxn vector of probabilities
        torch::Tensor predict_proba(const torch::Tensor&); // Return mxn tensor of probabilities
        double score(const vector<vector<int>>&, const vector<int>&);
        vector<string> topological_sort();
        vector<string> show();
        vector<string> graph(const string& title); // Returns a vector of strings representing the graph in graphviz format
        void initialize();
        void dump_cpt();
        inline string version() { return "0.1.0"; }
    };
}
#endif