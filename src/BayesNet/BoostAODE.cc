#include "BoostAODE.h"
#include "BayesMetrics.h"

namespace bayesnet {
    BoostAODE::BoostAODE() : Ensemble() {}
    void BoostAODE::buildModel(const torch::Tensor& weights)
    {
        models.clear();
        for (int i = 0; i < features.size(); ++i) {
            models.push_back(std::make_unique<SPODE>(i));
        }
    }
    void BoostAODE::trainModel(const torch::Tensor& weights)
    {
        // End building vectors
        Tensor weights_ = torch::full({ m }, 1.0 / m, torch::kDouble);
        auto X_ = dataset.index({ torch::indexing::Slice(0, dataset.size(0) - 1), "..." });
        auto featureSelection = metrics.SelectKBestWeighted(weights_, n); // Get all the features sorted
        for (int i = 0; i < features.size(); ++i) {
            models[i].fit(dataset, features, className, states, weights_);
            auto ypred = models[i].predict(X_);
            // em = np.sum(weights * (y_pred != self.y_)) / np.sum(weights)
            // am = np.log((1 - em) / em) + np.log(estimator.n_classes_ - 1)
            // # Step 3.2: Update weights for next classifier
            // weights = [
            //     wm * np.exp(am * (ym != yp))
            //         for wm, ym, yp in zip(weights, self.y_, y_pred)
            // ]
            // # Step 4: Add the new model
            // self.estimators_.append(estimator)
        }
    }
    vector<string> BoostAODE::graph(const string& title) const
    {
        return Ensemble::graph(title);
    }
}