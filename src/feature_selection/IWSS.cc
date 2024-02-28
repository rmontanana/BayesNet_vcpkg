#include "IWSS.h"
#include <limits>
#include "bayesnetUtils.h"
namespace bayesnet {
    IWSS::IWSS(const torch::Tensor& samples, const std::vector<std::string>& features, const std::string& className, const int maxFeatures, const int classNumStates, const torch::Tensor& weights, const double threshold) :
        FeatureSelect(samples, features, className, maxFeatures, classNumStates, weights), threshold(threshold)
    {
        if (threshold < 0 || threshold > .5) {
            throw std::invalid_argument("Threshold has to be in [0, 0.5]");
        }
    }
    void IWSS::fit()
    {
        initialize();
        computeSuLabels();
        auto featureOrder = argsort(suLabels); // sort descending order
        auto featureOrderCopy = featureOrder;
        // Add first and second features to result
        //     First with its own score
        auto first_feature = pop_first(featureOrderCopy);
        selectedFeatures.push_back(first_feature);
        selectedScores.push_back(suLabels.at(first_feature));
        //     Second with the score of the candidates
        selectedFeatures.push_back(pop_first(featureOrderCopy));
        auto merit = computeMeritCFS();
        selectedScores.push_back(merit);
        for (const auto feature : featureOrderCopy) {
            selectedFeatures.push_back(feature);
            // Compute merit with selectedFeatures
            auto meritNew = computeMeritCFS();
            double delta = merit != 0.0 ? std::abs(merit - meritNew) / merit : 0.0;
            if (meritNew > merit || delta < threshold) {
                if (meritNew > merit) {
                    merit = meritNew;
                }
                selectedScores.push_back(meritNew);
            } else {
                selectedFeatures.pop_back();
                break;
            }
            if (selectedFeatures.size() == maxFeatures) {
                break;
            }
        }
        fitted = true;
    }
}