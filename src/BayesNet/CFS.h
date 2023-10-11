#ifndef CFS_H
#define CFS_H
#include <torch/torch.h>
#include <vector>
#include "BayesMetrics.h"
using namespace std;
namespace bayesnet {
    class CFS : public Metrics {
    public:
        // dataset is a n+1xm tensor of integers where dataset[-1] is the y vector
        CFS(const torch::Tensor& samples, const vector<string>& features, const string& className, const int maxFeatures, const int classNumStates, const torch::Tensor& weights);
        virtual ~CFS() {};
        void fit();
        vector<int> getFeatures() const;
        vector<double> getScores() const;
    private:
        void computeSuLabels();
        double computeSuFeatures(const int a, const int b);
        double symmetricalUncertainty(int a, int b);
        double computeMerit();
        bool computeContinueCondition(const vector<int>& featureOrder);
        vector<pair<int, int>> combinations(const vector<int>& features);
        const torch::Tensor& weights;
        int maxFeatures;
        vector<int> cfsFeatures;
        vector<double> cfsScores;
        vector<double> suLabels;
        bool fitted = false;
    };
}
#endif