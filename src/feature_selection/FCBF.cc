#include "utils/bayesnetUtils.h"
#include "FCBF.h"
namespace bayesnet {

    FCBF::FCBF(const torch::Tensor& samples, const std::vector<std::string>& features, const std::string& className, const int maxFeatures, const int classNumStates, const torch::Tensor& weights, const double threshold) :
        FeatureSelect(samples, features, className, maxFeatures, classNumStates, weights), threshold(threshold)
    {
        if (threshold < 1e-7) {
            throw std::invalid_argument("Threshold cannot be less than 1e-7");
        }
    }
    void FCBF::fit()
    {
        initialize();
        computeSuLabels();
        auto featureOrder = argsort(suLabels); // sort descending order
        auto featureOrderCopy = featureOrder;
        for (const auto& feature : featureOrder) {
            // Don't self compare
            featureOrderCopy.erase(featureOrderCopy.begin());
            if (suLabels.at(feature) == 0.0) {
                // The feature has been removed from the list
                continue;
            }
            if (suLabels.at(feature) < threshold) {
                break;
            }
            // Remove redundant features
            for (const auto& featureCopy : featureOrderCopy) {
                double value = computeSuFeatures(feature, featureCopy);
                if (value >= suLabels.at(featureCopy)) {
                    // Remove feature from list
                    suLabels[featureCopy] = 0.0;
                }
            }
            selectedFeatures.push_back(feature);
            selectedScores.push_back(suLabels[feature]);
            if (selectedFeatures.size() == maxFeatures) {
                break;
            }
        }
        fitted = true;
    }
}