#include <set>
#include <functional>
#include <limits.h>
#include "BoostAODE.h"
#include "Colors.h"
#include "Folding.h"
#include "Paths.h"
#include "CFS.h"

namespace bayesnet {
    BoostAODE::BoostAODE() : Ensemble() {}
    void BoostAODE::buildModel(const torch::Tensor& weights)
    {
        // Models shall be built in trainModel
        models.clear();
        n_models = 0;
        // Prepare the validation dataset
        auto y_ = dataset.index({ -1, "..." });
        if (convergence) {
            // Prepare train & validation sets from train data
            auto fold = platform::StratifiedKFold(5, y_, 271);
            dataset_ = torch::clone(dataset);
            // save input dataset
            auto [train, test] = fold.getFold(0);
            auto train_t = torch::tensor(train);
            auto test_t = torch::tensor(test);
            // Get train and validation sets
            X_train = dataset.index({ torch::indexing::Slice(0, dataset.size(0) - 1), train_t });
            y_train = dataset.index({ -1, train_t });
            X_test = dataset.index({ torch::indexing::Slice(0, dataset.size(0) - 1), test_t });
            y_test = dataset.index({ -1, test_t });
            dataset = X_train;
            m = X_train.size(1);
            auto n_classes = states.at(className).size();
            metrics = Metrics(dataset, features, className, n_classes);
            // Build dataset with train data
            buildDataset(y_train);
        } else {
            // Use all data to train
            X_train = dataset.index({ torch::indexing::Slice(0, dataset.size(0) - 1), "..." });
            y_train = y_;
        }
    }
    void BoostAODE::setHyperparameters(nlohmann::json& hyperparameters)
    {
        // Check if hyperparameters are valid
        const vector<string> validKeys = { "repeatSparent", "maxModels", "ascending", "convergence", "cfs" };
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
        if (hyperparameters.contains("convergence")) {
            convergence = hyperparameters["convergence"];
        }
        if (hyperparameters.contains("cfs")) {
            cfs = hyperparameters["cfs"];
        }
    }
    unordered_set<int> BoostAODE::initializeModels()
    {
        unordered_set<int> featuresUsed;
        // Read the CFS features
        string output = "[", prefix = "";
        bool first = true;
        for (const auto& feature : features) {
            output += prefix + "'" + feature + "'";
            if (first) {
                prefix = ", ";
                first = false;
            }
        }
        output += "]";
        Tensor weights_ = torch::full({ m }, 1.0 / m, torch::kFloat64);
        int maxFeatures = 0;
        auto cfs = bayesnet::CFS(dataset, features, className, maxFeatures, states.at(className).size(), weights_);
        cfs.fit();
        auto cfsFeatures = cfs.getFeatures();
        for (const int& feature : cfsFeatures) {
            // cout << "Feature: [" << feature << "] " << feature << " " << features.at(feature) << endl;
            featuresUsed.insert(feature);
            unique_ptr<Classifier> model = std::make_unique<SPODE>(feature);
            model->fit(dataset, features, className, states, weights_);
            models.push_back(std::move(model));
            significanceModels.push_back(1.0);
            n_models++;
        }
        return featuresUsed;
    }
    void BoostAODE::trainModel(const torch::Tensor& weights)
    {
        unordered_set<int> featuresUsed;
        if (cfs) {
            featuresUsed = initializeModels();
        }
        if (maxModels == 0)
            maxModels = .1 * n > 10 ? .1 * n : n;
        Tensor weights_ = torch::full({ m }, 1.0 / m, torch::kFloat64);
        bool exitCondition = false;
        // Variables to control the accuracy finish condition
        double priorAccuracy = 0.0;
        double delta = 1.0;
        double threshold = 1e-4;
        int tolerance = 5; // number of times the accuracy can be lower than the threshold
        int count = 0; // number of times the accuracy is lower than the threshold
        fitted = true; // to enable predict
        // Step 0: Set the finish condition
        // if not repeatSparent a finish condition is run out of features
        // n_models == maxModels
        // epsiolon sub t > 0.5 => inverse the weights policy
        // validation error is not decreasing
        while (!exitCondition) {
            // Step 1: Build ranking with mutual information
            auto featureSelection = metrics.SelectKBestWeighted(weights_, ascending, n); // Get all the features sorted
            unique_ptr<Classifier> model;
            auto feature = featureSelection[0];
            if (!repeatSparent || featuresUsed.size() < featureSelection.size()) {
                bool used = true;
                for (const auto& feat : featureSelection) {
                    if (find(featuresUsed.begin(), featuresUsed.end(), feat) != featuresUsed.end()) {
                        continue;
                    }
                    used = false;
                    feature = feat;
                    break;
                }
                if (used) {
                    exitCondition = true;
                    continue;
                }
            }
            featuresUsed.insert(feature);
            model = std::make_unique<SPODE>(feature);
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
            n_models++;
            if (convergence) {
                auto y_val_predict = predict(X_test);
                double accuracy = (y_val_predict == y_test).sum().item<double>() / (double)y_test.size(0);
                if (priorAccuracy == 0) {
                    priorAccuracy = accuracy;
                } else {
                    delta = accuracy - priorAccuracy;
                }
                if (delta < threshold) {
                    count++;
                }
            }
            exitCondition = n_models >= maxModels && repeatSparent || epsilon_t > 0.5 || count > tolerance;
        }
        if (featuresUsed.size() != features.size()) {
            status = WARNING;
        }
    }
    vector<string> BoostAODE::graph(const string& title) const
    {
        return Ensemble::graph(title);
    }
}