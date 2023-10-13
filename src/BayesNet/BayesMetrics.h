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
        int classNumStates = 0;
        vector<double> scoresKBest;
        vector<int> featuresKBest; // sorted indices of the features
        double conditionalEntropy(const Tensor& firstFeature, const Tensor& secondFeature, const Tensor& weights);
    protected:
        Tensor samples; // n+1xm tensor used to fit the model where samples[-1] is the y vector
        string className;
        double entropy(const Tensor& feature, const Tensor& weights);
        vector<string> features;
        template <class T>
        vector<pair<T, T>> doCombinations(const vector<T>& source)
        {
            vector<pair<T, T>> result;
            for (int i = 0; i < source.size(); ++i) {
                T temp = source[i];
                for (int j = i + 1; j < source.size(); ++j) {
                    result.push_back({ temp, source[j] });
                }
            }
            return result;
        }
    public:
        Metrics() = default;
        Metrics(const torch::Tensor& samples, const vector<string>& features, const string& className, const int classNumStates);
        Metrics(const vector<vector<int>>& vsamples, const vector<int>& labels, const vector<string>& features, const string& className, const int classNumStates);
        vector<int> SelectKBestWeighted(const torch::Tensor& weights, bool ascending = false, unsigned k = 0);
        vector<double> getScoresKBest() const;
        double mutualInformation(const Tensor& firstFeature, const Tensor& secondFeature, const Tensor& weights);
        vector<float> conditionalEdgeWeights(vector<float>& weights); // To use in Python
        Tensor conditionalEdge(const torch::Tensor& weights);
        vector<pair<int, int>> maximumSpanningTree(const vector<string>& features, const Tensor& weights, const int root);
    };
}
#endif