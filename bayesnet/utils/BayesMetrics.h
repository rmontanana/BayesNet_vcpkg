// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef BAYESNET_METRICS_H
#define BAYESNET_METRICS_H
#include <vector>
#include <string>
#include <torch/torch.h>
namespace bayesnet {
    class Metrics {
    public:
        Metrics() = default;
        Metrics(const torch::Tensor& samples, const std::vector<std::string>& features, const std::string& className, const int classNumStates);
        Metrics(const std::vector<std::vector<int>>& vsamples, const std::vector<int>& labels, const std::vector<std::string>& features, const std::string& className, const int classNumStates);
        std::vector<int> SelectKBestWeighted(const torch::Tensor& weights, bool ascending = false, unsigned k = 0);
        std::vector<double> getScoresKBest() const;
        double mutualInformation(const torch::Tensor& firstFeature, const torch::Tensor& secondFeature, const torch::Tensor& weights);
        torch::Tensor conditionalEdge(const torch::Tensor& weights);
        std::vector<std::pair<int, int>> maximumSpanningTree(const std::vector<std::string>& features, const torch::Tensor& weights, const int root);
    protected:
        torch::Tensor samples; // n+1xm torch::Tensor used to fit the model where samples[-1] is the y std::vector
        std::string className;
        double entropy(const torch::Tensor& feature, const torch::Tensor& weights);
        std::vector<std::string> features;
        template <class T>
        std::vector<std::pair<T, T>> doCombinations(const std::vector<T>& source)
        {
            std::vector<std::pair<T, T>> result;
            for (int i = 0; i < source.size(); ++i) {
                T temp = source[i];
                for (int j = i + 1; j < source.size(); ++j) {
                    result.push_back({ temp, source[j] });
                }
            }
            return result;
        }
        template <class T>
        T pop_first(std::vector<T>& v)
        {
            T temp = v[0];
            v.erase(v.begin());
            return temp;
        }
    private:
        int classNumStates = 0;
        std::vector<double> scoresKBest;
        std::vector<int> featuresKBest; // sorted indices of the features
        double conditionalEntropy(const torch::Tensor& firstFeature, const torch::Tensor& secondFeature, const torch::Tensor& weights);
    };
}
#endif