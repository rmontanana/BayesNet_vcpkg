#ifndef NETWORK_H
#define NETWORK_H
#include "Node.h"
#include <map>
#include <vector>


namespace bayesnet {
    class Network {
    private:
        map<string, Node*> nodes;
        map<string, vector<int>> dataset;
        Node* root;
        int classNumStates;
        vector<string> features;
        string className;
        int laplaceSmoothing;
        bool isCyclic(const std::string&, std::unordered_set<std::string>&, std::unordered_set<std::string>&);
        pair<int, double> predict_sample(const vector<int>&);
        vector<double> exactInference(map<string, int>&);
        double computeFactor(map<string, int>&);
    public:
        Network();
        Network(int);
        Network(Network&);
        ~Network();
        void addNode(string, int);
        void addEdge(const string, const string);
        map<string, Node*>& getNodes();
        vector<string> getFeatures();
        int getClassNumStates();
        string getClassName();
        void fit(const vector<vector<int>>&, const vector<int>&, const vector<string>&, const string&);
        void estimateParameters();
        void setRoot(string);
        Node* getRoot();
        vector<int> predict(const vector<vector<int>>&);
        vector<pair<int, double>> predict_proba(const vector<vector<int>>&);
        double score(const vector<vector<int>>&, const vector<int>&);
    };
}
#endif