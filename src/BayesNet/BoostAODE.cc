#include "BoostAODE.h"
#include <set>
#include "BayesMetrics.h"
#include "Colors.h"
#include "Folding.h"

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
    void BoostAODE::validationInit()
    {
        auto y_ = dataset.index({ -1, "..." });
        auto fold = platform::StratifiedKFold(5, y_, 271);
        // save input dataset
        dataset_ = torch::clone(dataset);
        auto [train, test] = fold.getFold(0);
        auto train_t = torch::tensor(train);
        auto test_t = torch::tensor(test);
        // Get train and validation sets
        X_train = dataset.index({ "...", train_t });
        y_train = dataset.index({ -1, train_t });
        X_test = dataset.index({ "...", test_t });
        y_test = dataset.index({ -1, test_t });
        // Build dataset with train data
        dataset = X_train;
        buildDataset(y_train);
        m = X_train.size(1);
        auto n_classes = states.at(className).size();
        metrics = Metrics(dataset, features, className, n_classes);
    }
    void BoostAODE::trainModel(const torch::Tensor& weights)
    {
        models.clear();
        n_models = 0;
        if (maxModels == 0)
            maxModels = .1 * n > 10 ? .1 * n : n;
        validationInit();
        Tensor weights_ = torch::full({ m }, 1.0 / m, torch::kFloat64);
        bool exitCondition = false;
        unordered_set<int> featuresUsed;
        // Step 0: Set the finish condition
        // if not repeatSparent a finish condition is run out of features
        // n_models == maxModels
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
            auto ypred = model->predict(X_train);
            // Step 3.1: Compute the classifier amout of say
            auto mask_wrong = ypred != y_train;
            auto mask_right = ypred == y_train;
            auto masked_weights = weights_ * mask_wrong.to(weights_.dtype());
            double epsilon_t = masked_weights.sum().item<double>();
            double wt = (1 - epsilon_t) / epsilon_t;
            double alpha_t = epsilon_t == 0 ? 1 : 0.5 * log(wt);
            // Step 3.2: Update weights for next classifier
            // Step 3.2.1: Update weights of wrong samples
            weights_ += mask_wrong.to(weights_.dtype()) * exp(alpha_t) * weights_;
            // Step 3.2.2: Update weights of right samples
            weights_ += mask_right.to(weights_.dtype()) * exp(-alpha_t) * weights_;
            // Step 3.3: Normalise the weights
            double totalWeights = torch::sum(weights_).item<double>();
            weights_ = weights_ / totalWeights;
            // Step 3.4: Store classifier and its accuracy to weigh its future vote
            models.push_back(std::move(model));
            significanceModels.push_back(alpha_t);
            exitCondition = n_models == maxModels && repeatSparent || epsilon_t > 0.5;
        }
        if (featuresUsed.size() != features.size()) {
            status = WARNING;
        }
        weights.copy_(weights_);
    }
    vector<string> BoostAODE::graph(const string& title) const
    {
        return Ensemble::graph(title);
    }
}