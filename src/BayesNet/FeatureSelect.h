#ifndef FEATURE_SELECT_H
#define FEATURE_SELECT_H
#include <torch/torch.h>
#include <vector>
#include "BayesMetrics.h"
using namespace std;
namespace bayesnet {
    class FeatureSelect : public Metrics {
    public:
        // dataset is a n+1xm tensor of integers where dataset[-1] is the y vector
        FeatureSelect(const torch::Tensor& samples, const vector<string>& features, const string& className, const int maxFeatures, const int classNumStates, const torch::Tensor& weights);
        virtual ~FeatureSelect() {};
        virtual void fit() = 0;
        vector<int> getFeatures() const;
        vector<double> getScores() const;
    protected:
        void computeSuLabels();
        double computeSuFeatures(const int a, const int b);
        double symmetricalUncertainty(int a, int b);
        double computeMeritCFS();
        const torch::Tensor& weights;
        int maxFeatures;
        vector<int> selectedFeatures;
        vector<double> selectedScores;
        vector<double> suLabels;
        map<pair<int, int>, double> suFeatures;
        bool fitted = false;
    };
}
#endif