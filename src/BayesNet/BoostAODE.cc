#include <set>
#include <functional>
#include <limits.h>
#include "BoostAODE.h"
#include "CFS.h"
#include "FCBF.h"
#include "IWSS.h"
#include "folding.hpp"

namespace bayesnet {
    BoostAODE::BoostAODE() : Ensemble()
    {
        validHyperparameters = { "repeatSparent", "maxModels", "ascending", "convergence", "threshold", "select_features", "tolerance" };

    }
    void BoostAODE::buildModel(const torch::Tensor& weights)
    {
        // Models shall be built in trainModel
        models.clear();
        n_models = 0;
        // Prepare the validation dataset
        auto y_ = dataset.index({ -1, "..." });
        if (convergence) {
            // Prepare train & validation sets from train data
            auto fold = folding::StratifiedKFold(5, y_, 271);
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
    void BoostAODE::setHyperparameters(const nlohmann::json& hyperparameters_)
    {
        auto hyperparameters = hyperparameters_;
        if (hyperparameters.contains("repeatSparent")) {
            repeatSparent = hyperparameters["repeatSparent"];
            hyperparameters.erase("repeatSparent");
        }
        if (hyperparameters.contains("maxModels")) {
            maxModels = hyperparameters["maxModels"];
            hyperparameters.erase("maxModels");
        }
        if (hyperparameters.contains("ascending")) {
            ascending = hyperparameters["ascending"];
            hyperparameters.erase("ascending");
        }
        if (hyperparameters.contains("convergence")) {
            convergence = hyperparameters["convergence"];
            hyperparameters.erase("convergence");
        }
        if (hyperparameters.contains("threshold")) {
            threshold = hyperparameters["threshold"];
            hyperparameters.erase("threshold");
        }
        if (hyperparameters.contains("tolerance")) {
            tolerance = hyperparameters["tolerance"];
            hyperparameters.erase("tolerance");
        }
        if (hyperparameters.contains("select_features")) {
            auto selectedAlgorithm = hyperparameters["select_features"];
            std::vector<std::string> algos = { "IWSS", "FCBF", "CFS" };
            selectFeatures = true;
            algorithm = selectedAlgorithm;
            if (std::find(algos.begin(), algos.end(), selectedAlgorithm) == algos.end()) {
                throw std::invalid_argument("Invalid selectFeatures value [IWSS, FCBF, CFS]");
            }
            hyperparameters.erase("select_features");
        }
        if (!hyperparameters.empty()) {
            throw std::invalid_argument("Invalid hyperparameters" + hyperparameters.dump());
        }
    }
    std::unordered_set<int> BoostAODE::initializeModels()
    {
        std::unordered_set<int> featuresUsed;
        torch::Tensor weights_ = torch::full({ m }, 1.0 / m, torch::kFloat64);
        int maxFeatures = 0;
        if (algorithm == "CFS") {
            featureSelector = new CFS(dataset, features, className, maxFeatures, states.at(className).size(), weights_);
        } else if (algorithm == "IWSS") {
            if (threshold < 0 || threshold >0.5) {
                throw std::invalid_argument("Invalid threshold value for IWSS [0, 0.5]");
            }
            featureSelector = new IWSS(dataset, features, className, maxFeatures, states.at(className).size(), weights_, threshold);
        } else if (algorithm == "FCBF") {
            if (threshold < 1e-7 || threshold > 1) {
                throw std::invalid_argument("Invalid threshold value [1e-7, 1]");
            }
            featureSelector = new FCBF(dataset, features, className, maxFeatures, states.at(className).size(), weights_, threshold);
        }
        featureSelector->fit();
        auto cfsFeatures = featureSelector->getFeatures();
        for (const int& feature : cfsFeatures) {
            // std::cout << "Feature: [" << feature << "] " << feature << " " << features.at(feature) << std::endl;
            featuresUsed.insert(feature);
            std::unique_ptr<Classifier> model = std::make_unique<SPODE>(feature);
            model->fit(dataset, features, className, states, weights_);
            models.push_back(std::move(model));
            significanceModels.push_back(1.0);
            n_models++;
        }
        notes.push_back("Used features in initialization: " + std::to_string(featuresUsed.size()) + " of " + std::to_string(features.size()) + " with " + algorithm);
        delete featureSelector;
        return featuresUsed;
    }
    void BoostAODE::trainModel(const torch::Tensor& weights)
    {
        std::unordered_set<int> featuresUsed;
        if (selectFeatures) {
            featuresUsed = initializeModels();
        }
        if (maxModels == 0)
            maxModels = .1 * n > 10 ? .1 * n : n;
        torch::Tensor weights_ = torch::full({ m }, 1.0 / m, torch::kFloat64);
        bool exitCondition = false;
        // Variables to control the accuracy finish condition
        double priorAccuracy = 0.0;
        double delta = 1.0;
        double threshold = 1e-4;
        int count = 0; // number of times the accuracy is lower than the threshold
        fitted = true; // to enable predict
        // Step 0: Set the finish condition
        // if not repeatSparent a finish condition is run out of features
        // n_models == maxModels
        // epsilon sub t > 0.5 => inverse the weights policy
        // validation error is not decreasing
        while (!exitCondition) {
            // Step 1: Build ranking with mutual information
            auto featureSelection = metrics.SelectKBestWeighted(weights_, ascending, n); // Get all the features sorted
            std::unique_ptr<Classifier> model;
            auto feature = featureSelection[0];
            if (!repeatSparent || featuresUsed.size() < featureSelection.size()) {
                bool used = true;
                for (const auto& feat : featureSelection) {
                    if (std::find(featuresUsed.begin(), featuresUsed.end(), feat) != featuresUsed.end()) {
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
            notes.push_back("Used features in train: " + std::to_string(featuresUsed.size()) + " of " + std::to_string(features.size()));
            status = WARNING;
        }
    }
    std::vector<std::string> BoostAODE::graph(const std::string& title) const
    {
        return Ensemble::graph(title);
    }
}