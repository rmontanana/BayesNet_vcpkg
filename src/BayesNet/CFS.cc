#include "CFS.h"
#include <limits>
#include "bayesnetUtils.h"
namespace bayesnet {
    CFS::CFS(const torch::Tensor& samples, const vector<string>& features, const string& className, const int maxFeatures, const int classNumStates, const torch::Tensor& weights) :
        Metrics(samples, features, className, classNumStates), maxFeatures(maxFeatures == 0 ? samples.size(0) - 1 : maxFeatures), weights(weights)

    {
    }
    double CFS::symmetricalUncertainty(int a, int b)
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
    void CFS::computeSuLabels()
    {
        // Compute Simmetrical Uncertainty between features and labels
        // https://en.wikipedia.org/wiki/Symmetric_uncertainty
        for (int i = 0; i < features.size(); ++i) {
            suLabels.push_back(symmetricalUncertainty(i, -1));
        }

    }
    double CFS::computeSuFeatures(const int firstFeature, const int secondFeature)
    {
        // Compute Simmetrical Uncertainty between features
        // https://en.wikipedia.org/wiki/Symmetric_uncertainty
        try {
            return suFeatures.at({ firstFeature, secondFeature });
        }
        catch (const out_of_range& e) {
            auto result = symmetricalUncertainty(firstFeature, secondFeature);
            suFeatures[{firstFeature, secondFeature}] = result;
            return result;
        }
    }
    double CFS::computeMerit()
    {
        double result;
        double rcf = 0;
        for (auto feature : cfsFeatures) {
            rcf += suLabels[feature];
        }
        double rff = 0;
        int n = cfsFeatures.size();
        for (const auto& item : doCombinations(cfsFeatures)) {
            rff += computeSuFeatures(item.first, item.second);
        }
        return rcf / sqrt(n + (n * n - n) * rff);
    }
    void CFS::fit()
    {
        cfsFeatures.clear();
        computeSuLabels();
        auto featureOrder = argsort(suLabels); // sort descending order
        auto continueCondition = true;
        auto feature = featureOrder[0];
        cfsFeatures.push_back(feature);
        cfsScores.push_back(suLabels[feature]);
        cfsFeatures.erase(cfsFeatures.begin());
        while (continueCondition) {
            double merit = numeric_limits<double>::lowest();
            int bestFeature = -1;
            for (auto feature : featureOrder) {
                cfsFeatures.push_back(feature);
                auto meritNew = computeMerit(); // Compute merit with cfsFeatures
                if (meritNew > merit) {
                    merit = meritNew;
                    bestFeature = feature;
                }
                cfsFeatures.pop_back();
            }
            if (bestFeature == -1) {
                // meritNew has to be nan due to constant features
                break;
            }
            cfsFeatures.push_back(bestFeature);
            cfsScores.push_back(merit);
            featureOrder.erase(remove(featureOrder.begin(), featureOrder.end(), bestFeature), featureOrder.end());
            continueCondition = computeContinueCondition(featureOrder);
        }
        fitted = true;
    }
    bool CFS::computeContinueCondition(const vector<int>& featureOrder)
    {
        if (cfsFeatures.size() == maxFeatures || featureOrder.size() == 0) {
            return false;
        }
        if (cfsScores.size() >= 5) {
            /*
            "To prevent the best first search from exploring the entire
            feature subset search space, a stopping criterion is imposed.
            The search will terminate if five consecutive fully expanded
            subsets show no improvement over the current best subset."
            as stated in Mark A.Hall Thesis
            */
            double item_ant = numeric_limits<double>::lowest();
            int num = 0;
            vector<double> lastFive(cfsScores.end() - 5, cfsScores.end());
            for (auto item : lastFive) {
                if (item_ant == numeric_limits<double>::lowest()) {
                    item_ant = item;
                }
                if (item > item_ant) {
                    break;
                } else {
                    num++;
                    item_ant = item;
                }
            }
            if (num == 5) {
                return false;
            }
        }
        return true;
    }
    vector<int> CFS::getFeatures() const
    {
        if (!fitted) {
            throw runtime_error("CFS not fitted");
        }
        return cfsFeatures;
    }
    vector<double> CFS::getScores() const
    {
        if (!fitted) {
            throw runtime_error("CFS not fitted");
        }
        return cfsScores;
    }
}