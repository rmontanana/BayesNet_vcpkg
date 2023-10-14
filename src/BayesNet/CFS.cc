#include "CFS.h"
#include <limits>
#include "bayesnetUtils.h"
namespace bayesnet {
    void CFS::fit()
    {
        initialize();
        computeSuLabels();
        auto featureOrder = argsort(suLabels); // sort descending order
        auto continueCondition = true;
        auto feature = featureOrder[0];
        selectedFeatures.push_back(feature);
        selectedScores.push_back(suLabels[feature]);
        selectedFeatures.erase(selectedFeatures.begin());
        while (continueCondition) {
            double merit = numeric_limits<double>::lowest();
            int bestFeature = -1;
            for (auto feature : featureOrder) {
                selectedFeatures.push_back(feature);
                // Compute merit with selectedFeatures
                auto meritNew = computeMeritCFS();
                if (meritNew > merit) {
                    merit = meritNew;
                    bestFeature = feature;
                }
                selectedFeatures.pop_back();
            }
            if (bestFeature == -1) {
                // meritNew has to be nan due to constant features
                break;
            }
            selectedFeatures.push_back(bestFeature);
            selectedScores.push_back(merit);
            featureOrder.erase(remove(featureOrder.begin(), featureOrder.end(), bestFeature), featureOrder.end());
            continueCondition = computeContinueCondition(featureOrder);
        }
        fitted = true;
    }
    bool CFS::computeContinueCondition(const vector<int>& featureOrder)
    {
        if (selectedFeatures.size() == maxFeatures || featureOrder.size() == 0) {
            return false;
        }
        if (selectedScores.size() >= 5) {
            /*
            "To prevent the best first search from exploring the entire
            feature subset search space, a stopping criterion is imposed.
            The search will terminate if five consecutive fully expanded
            subsets show no improvement over the current best subset."
            as stated in Mark A.Hall Thesis
            */
            double item_ant = numeric_limits<double>::lowest();
            int num = 0;
            vector<double> lastFive(selectedScores.end() - 5, selectedScores.end());
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
}