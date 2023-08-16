#include "BoostAODE.h"
#include "FeatureSelect.h"

namespace bayesnet {
    BoostAODE::BoostAODE() : Ensemble() {}
    void BoostAODE::buildModel(const torch::Tensor& weights)
    {
        models.clear();
        int n_samples = dataset.size(1);
        int n_features = dataset.size(0);
        features::samples_t vsamples;
        for (auto i = 0; i < n_samples; ++i) {
            auto row = dataset.index({ "...", i });
            // convert row to std::vector<int>
            auto vrow = vector<int>(row.data_ptr<int>(), row.data_ptr<int>() + row.numel());
            vsamples.push_back(vrow);
        }
        auto vweights = features::weights_t(n_samples, 1.0 / n_samples);
        auto row = dataset.index({ -1, "..." });
        auto yv = features::labels_t(row.data_ptr<int>(), row.data_ptr<int>() + row.numel());
        auto featureSelection = features::SelectKBestWeighted(vsamples, yv, vweights, n_features, true);
        auto features = featureSelection.fit().getFeatures();
        // features = (
        //     CSelectKBestWeighted(
        //         self.X_, self.y_, weights, k = self.n_features_in_
        //     )
        //     .fit()
        //     .get_features()
        auto scores = features::score_t(n_features, 0.0);
        for (int i = 0; i < features.size(); ++i) {
            models.push_back(std::make_unique<SPODE>(i));
        }
    }
    vector<string> BoostAODE::graph(const string& title) const
    {
        return Ensemble::graph(title);
    }
}