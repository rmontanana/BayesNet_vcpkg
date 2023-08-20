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
        vector<double> scoresKBest;
        vector<int> featuresKBest; // sorted indices of the features
        double entropy(const Tensor& feature, const Tensor& weights);
        double conditionalEntropy(const Tensor& firstFeature, const Tensor& secondFeature, const Tensor& weights);
        vector<pair<string, string>> doCombinations(const vector<string>&);
    public:
        Metrics() = default;
        Metrics(const torch::Tensor& samples, const vector<string>& features, const string& className, const int classNumStates);
        Metrics(const vector<vector<int>>& vsamples, const vector<int>& labels, const vector<string>& features, const string& className, const int classNumStates);
        vector<int> SelectKBestWeighted(const torch::Tensor& weights, bool ascending=false, unsigned k = 0);
        vector<double> getScoresKBest() const;
        double mutualInformation(const Tensor& firstFeature, const Tensor& secondFeature, const Tensor& weights);
        vector<float> conditionalEdgeWeights(vector<float>& weights); // To use in Python
        Tensor conditionalEdge(const torch::Tensor& weights);
        vector<pair<int, int>> maximumSpanningTree(const vector<string>& features, const Tensor& weights, const int root);
    };
}
#endif