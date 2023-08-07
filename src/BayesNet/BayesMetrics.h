#ifndef BAYESNET_METRICS_H
#define BAYESNET_METRICS_H
#include <torch/torch.h>
#include <vector>
#include <string>
namespace bayesnet {
    using namespace std;
    using namespace torch;
    class Metrics {
    private:
        Tensor samples; // nxm tensor used to fit the model
        vector<string> features;
        string className;
        int classNumStates = 0;
    public:
        Metrics() = default;
        Metrics(const Tensor&, const vector<string>&, const string&, const int);
        Metrics(const vector<vector<int>>&, const vector<int>&, const vector<string>&, const string&, const int);
        double entropy(const Tensor&);
        double conditionalEntropy(const Tensor&, const Tensor&);
        double mutualInformation(const Tensor&, const Tensor&);
        vector<float> conditionalEdgeWeights(); // To use in Python
        Tensor conditionalEdge();
        vector<pair<string, string>> doCombinations(const vector<string>&);
        vector<pair<int, int>> maximumSpanningTree(const vector<string>& features, const Tensor& weights, const int root);
    };
}
#endif