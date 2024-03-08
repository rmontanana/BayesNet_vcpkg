#ifndef FEATURE_SELECT_H
#define FEATURE_SELECT_H
#include <torch/torch.h>
#include <vector>
#include "bayesnet/utils/BayesMetrics.h"
namespace bayesnet {
    class FeatureSelect : public Metrics {
    public:
        // dataset is a n+1xm tensor of integers where dataset[-1] is the y std::vector
        FeatureSelect(const torch::Tensor& samples, const std::vector<std::string>& features, const std::string& className, const int maxFeatures, const int classNumStates, const torch::Tensor& weights);
        virtual ~FeatureSelect() {};
        virtual void fit() = 0;
        std::vector<int> getFeatures() const;
        std::vector<double> getScores() const;
    protected:
        void initialize();
        void computeSuLabels();
        double computeSuFeatures(const int a, const int b);
        double symmetricalUncertainty(int a, int b);
        double computeMeritCFS();
        const torch::Tensor& weights;
        int maxFeatures;
        std::vector<int> selectedFeatures;
        std::vector<double> selectedScores;
        std::vector<double> suLabels;
        std::map<std::pair<int, int>, double> suFeatures;
        bool fitted = false;
    };
}
#endif