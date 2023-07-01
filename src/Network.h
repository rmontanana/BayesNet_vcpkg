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
        vector<string> features;
        string className;
        int laplaceSmoothing;
        bool isCyclic(const std::string&, std::unordered_set<std::string>&, std::unordered_set<std::string>&);
    public:
        Network();
        Network(int);
        ~Network();
        void addNode(string, int);
        void addEdge(const string, const string);
        map<string, Node*>& getNodes();
        void fit(const vector<vector<int>>&, const vector<int>&, const vector<string>&, const string&);
        void estimateParameters();
        void buildNetwork();
        void setRoot(string);
        Node* getRoot();
    };
}
#endif