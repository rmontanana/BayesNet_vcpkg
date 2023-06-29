#ifndef NETWORK_H
#define NETWORK_H
#include "Node.h"
#include <map>
#include <vector>
namespace bayesnet {
    class Network {
    private:
        map<string, Node*> nodes;
        map<string, torch::Tensor> cpds;  // Map from CPD key to CPD tensor
    public:
        ~Network();
        void addNode(string, int);
        void addEdge(const string, const string);
        map<string, Node*>& getNodes();
        void fit(const vector<vector<int>>&, const int);
        torch::Tensor& getCPD(const string&);
        void setCPD(const string&, const torch::Tensor&);
    };
}
#endif