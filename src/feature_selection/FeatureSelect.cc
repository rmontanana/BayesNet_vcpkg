#include "FeatureSelect.h"
#include <limits>
#include "bayesnetUtils.h"
namespace bayesnet {
    FeatureSelect::FeatureSelect(const torch::Tensor& samples, const std::vector<std::string>& features, const std::string& className, const int maxFeatures, const int classNumStates, const torch::Tensor& weights) :
        Metrics(samples, features, className, classNumStates), maxFeatures(maxFeatures == 0 ? samples.size(0) - 1 : maxFeatures), weights(weights)

    {
    }
    void FeatureSelect::initialize()
    {
        selectedFeatures.clear();
        selectedScores.clear();
    }
    double FeatureSelect::symmetricalUncertainty(int a, int b)
    {
        /*
        Compute symmetrical uncertainty. Normalize* information gain (mutual
        information) with the entropies of the features in order to compensate
        the bias due to high cardinality features. *Range [0, 1]
        (https://www.sciencedirect.com/science/article/pii/S0020025519303603)
        */
        auto x = samples.index({ a, "..." });
        auto y = samples.index({ b, "..." });
        auto mu = mutualInformation(x, y, weights);
        auto hx = entropy(x, weights);
        auto hy = entropy(y, weights);
        return 2.0 * mu / (hx + hy);
    }
    void FeatureSelect::computeSuLabels()
    {
        // Compute Simmetrical Uncertainty between features and labels
        // https://en.wikipedia.org/wiki/Symmetric_uncertainty
        for (int i = 0; i < features.size(); ++i) {
            suLabels.push_back(symmetricalUncertainty(i, -1));
        }
    }
    double FeatureSelect::computeSuFeatures(const int firstFeature, const int secondFeature)
    {
        // Compute Simmetrical Uncertainty between features
        // https://en.wikipedia.org/wiki/Symmetric_uncertainty
        try {
            return suFeatures.at({ firstFeature, secondFeature });
        }
        catch (const std::out_of_range& e) {
            double result = symmetricalUncertainty(firstFeature, secondFeature);
            suFeatures[{firstFeature, secondFeature}] = result;
            return result;
        }
    }
    double FeatureSelect::computeMeritCFS()
    {
        double rcf = 0;
        for (auto feature : selectedFeatures) {
            rcf += suLabels[feature];
        }
        double rff = 0;
        int n = selectedFeatures.size();
        for (const auto& item : doCombinations(selectedFeatures)) {
            rff += computeSuFeatures(item.first, item.second);
        }
        return rcf / sqrt(n + (n * n - n) * rff);
    }
    std::vector<int> FeatureSelect::getFeatures() const
    {
        if (!fitted) {
            throw std::runtime_error("FeatureSelect not fitted");
        }
        return selectedFeatures;
    }
    std::vector<double> FeatureSelect::getScores() const
    {
        if (!fitted) {
            throw std::runtime_error("FeatureSelect not fitted");
        }
        return selectedScores;
    }
}