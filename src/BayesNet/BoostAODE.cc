#include "BoostAODE.h"
#include <set>
#include "BayesMetrics.h"

namespace bayesnet {
    BoostAODE::BoostAODE() : Ensemble() {}
    void BoostAODE::buildModel(const torch::Tensor& weights)
    {
        // Models shall be built in trainModel
    }
    void BoostAODE::setHyperparameters(nlohmann::json& hyperparameters)
    {
        // Check if hyperparameters are valid
        const vector<string> validKeys = { "repeatSparent", "maxModels", "ascending" };
        checkHyperparameters(validKeys, hyperparameters);
        if (hyperparameters.contains("repeatSparent")) {
            repeatSparent = hyperparameters["repeatSparent"];
        }
        if (hyperparameters.contains("maxModels")) {
            maxModels = hyperparameters["maxModels"];
        }
        if (hyperparameters.contains("ascending")) {
            ascending = hyperparameters["ascending"];
        }
    }
    void BoostAODE::trainModel(const torch::Tensor& weights)
    {
        models.clear();
        n_models = 0;
        if (maxModels == 0)
            maxModels = .1 * n > 10 ? .1 * n : n;
        Tensor weights_ = torch::full({ m }, 1.0 / m, torch::kFloat64);
        auto X_ = dataset.index({ torch::indexing::Slice(0, dataset.size(0) - 1), "..." });
        auto y_ = dataset.index({ -1, "..." });
        bool exitCondition = false;
        unordered_set<int> featuresUsed;
        // Step 0: Set the finish condition
        // if not repeatSparent a finish condition is run out of features
        // n_models == maxModels
        int numClasses = states[className].size();
        while (!exitCondition) {
            // Step 1: Build ranking with mutual information
            auto featureSelection = metrics.SelectKBestWeighted(weights_, ascending, n); // Get all the features sorted
            unique_ptr<Classifier> model;
            auto feature = featureSelection[0];
            if (!repeatSparent || featuresUsed.size() < featureSelection.size()) {
                bool found = false;
                for (auto feat : featureSelection) {
                    if (find(featuresUsed.begin(), featuresUsed.end(), feat) != featuresUsed.end()) {
                        continue;
                    }
                    found = true;
                    feature = feat;
                    break;
                }
                if (!found) {
                    exitCondition = true;
                    continue;
                }
            }
            featuresUsed.insert(feature);
            model = std::make_unique<SPODE>(feature);
            n_models++;
            model->fit(dataset, features, className, states, weights_);
            auto ypred = model->predict(X_);
            // Step 3.1: Compute the classifier amout of say
            auto mask_wrong = ypred != y_;
            auto masked_weights = weights_ * mask_wrong.to(weights_.dtype());
            double wrongWeights = masked_weights.sum().item<double>();
            double significance = wrongWeights == 0 ? 1 : 0.5 * log((1 - wrongWeights) / wrongWeights);
            // Step 3.2: Update weights for next classifier
            // Step 3.2.1: Update weights of wrong samples
            weights_ += mask_wrong.to(weights_.dtype()) * exp(significance) * weights_;
            // Step 3.3: Normalise the weights
            double totalWeights = torch::sum(weights_).item<double>();
            weights_ = weights_ / totalWeights;
            // Step 3.4: Store classifier and its accuracy to weigh its future vote
            models.push_back(std::move(model));
            significanceModels.push_back(significance);
            exitCondition = n_models == maxModels && repeatSparent;
        }
        if (featuresUsed.size() != features.size()) {
            cout << "Warning: BoostAODE did not use all the features" << endl;
        }
        weights.copy_(weights_);
    }
    vector<string> BoostAODE::graph(const string& title) const
    {
        return Ensemble::graph(title);
    }
}