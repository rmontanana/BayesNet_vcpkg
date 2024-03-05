#include <set>
#include <functional>
#include <limits.h>
#include <tuple>
#include "BoostAODE.h"
#include "CFS.h"
#include "FCBF.h"
#include "IWSS.h"
#include "folding.hpp"

namespace bayesnet {
    struct {
        std::string CFS = "CFS";
        std::string FCBF = "FCBF";
        std::string IWSS = "IWSS";
    }SelectFeatures;
    struct {
        std::string ASC = "asc";
        std::string DESC = "desc";
        std::string RAND = "rand";
    }Orders;
    BoostAODE::BoostAODE(bool predict_voting) : Ensemble(predict_voting)
    {
        validHyperparameters = {
            "repeatSparent", "maxModels", "order", "convergence", "threshold",
            "select_features", "tolerance", "predict_voting", "predict_single"
        };

    }
    void BoostAODE::buildModel(const torch::Tensor& weights)
    {
        // Models shall be built in trainModel
        models.clear();
        significanceModels.clear();
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
        if (hyperparameters.contains("order")) {
            std::vector<std::string> algos = { Orders.ASC, Orders.DESC, Orders.RAND };
            order_algorithm = hyperparameters["order"];
            if (std::find(algos.begin(), algos.end(), order_algorithm) == algos.end()) {
                throw std::invalid_argument("Invalid order algorithm, valid values [" + Orders.ASC + ", " + Orders.DESC + ", " + Orders.RAND + "]");
            }
            hyperparameters.erase("order");
        }
        if (hyperparameters.contains("convergence")) {
            convergence = hyperparameters["convergence"];
            hyperparameters.erase("convergence");
        }
        if (hyperparameters.contains("predict_single")) {
            predict_single = hyperparameters["predict_single"];
            hyperparameters.erase("predict_single");
        }
        if (hyperparameters.contains("threshold")) {
            threshold = hyperparameters["threshold"];
            hyperparameters.erase("threshold");
        }
        if (hyperparameters.contains("tolerance")) {
            tolerance = hyperparameters["tolerance"];
            hyperparameters.erase("tolerance");
        }
        if (hyperparameters.contains("predict_voting")) {
            predict_voting = hyperparameters["predict_voting"];
            hyperparameters.erase("predict_voting");
        }
        if (hyperparameters.contains("select_features")) {
            auto selectedAlgorithm = hyperparameters["select_features"];
            std::vector<std::string> algos = { SelectFeatures.IWSS, SelectFeatures.CFS, SelectFeatures.CFS };
            selectFeatures = true;
            select_features_algorithm = selectedAlgorithm;
            if (std::find(algos.begin(), algos.end(), selectedAlgorithm) == algos.end()) {
                throw std::invalid_argument("Invalid selectFeatures value, valid values [" + SelectFeatures.IWSS + ", " + SelectFeatures.CFS + ", " + SelectFeatures.FCBF + "]");
            }
            hyperparameters.erase("select_features");
        }
        if (!hyperparameters.empty()) {
            throw std::invalid_argument("Invalid hyperparameters" + hyperparameters.dump());
        }
    }
    std::tuple<torch::Tensor&, double, bool> update_weights(torch::Tensor& ytrain, torch::Tensor& ypred, torch::Tensor& weights)
    {
        bool terminate = false;
        double alpha_t = 0;
        auto mask_wrong = ypred != ytrain;
        auto mask_right = ypred == ytrain;
        auto masked_weights = weights * mask_wrong.to(weights.dtype());
        double epsilon_t = masked_weights.sum().item<double>();
        if (epsilon_t > 0.5) {
            // Inverse the weights policy (plot ln(wt))
            // "In each round of AdaBoost, there is a sanity check to ensure that the current base 
            // learner is better than random guess" (Zhi-Hua Zhou, 2012)
            terminate = true;
        } else {
            double wt = (1 - epsilon_t) / epsilon_t;
            alpha_t = epsilon_t == 0 ? 1 : 0.5 * log(wt);
            // Step 3.2: Update weights for next classifier
            // Step 3.2.1: Update weights of wrong samples
            weights += mask_wrong.to(weights.dtype()) * exp(alpha_t) * weights;
            // Step 3.2.2: Update weights of right samples
            weights += mask_right.to(weights.dtype()) * exp(-alpha_t) * weights;
            // Step 3.3: Normalise the weights
            double totalWeights = torch::sum(weights).item<double>();
            weights = weights / totalWeights;
        }
        return { weights, alpha_t, terminate };
    }
    std::unordered_set<int> BoostAODE::initializeModels()
    {
        std::unordered_set<int> featuresUsed;
        torch::Tensor weights_ = torch::full({ m }, 1.0 / m, torch::kFloat64);
        int maxFeatures = 0;
        if (select_features_algorithm == SelectFeatures.CFS) {
            featureSelector = new CFS(dataset, features, className, maxFeatures, states.at(className).size(), weights_);
        } else if (select_features_algorithm == SelectFeatures.IWSS) {
            if (threshold < 0 || threshold >0.5) {
                throw std::invalid_argument("Invalid threshold value for " + SelectFeatures.IWSS + " [0, 0.5]");
            }
            featureSelector = new IWSS(dataset, features, className, maxFeatures, states.at(className).size(), weights_, threshold);
        } else if (select_features_algorithm == SelectFeatures.FCBF) {
            if (threshold < 1e-7 || threshold > 1) {
                throw std::invalid_argument("Invalid threshold value for " + SelectFeatures.FCBF + " [1e-7, 1]");
            }
            featureSelector = new FCBF(dataset, features, className, maxFeatures, states.at(className).size(), weights_, threshold);
        }
        featureSelector->fit();
        auto cfsFeatures = featureSelector->getFeatures();
        for (const int& feature : cfsFeatures) {
            featuresUsed.insert(feature);
            std::unique_ptr<Classifier> model = std::make_unique<SPODE>(feature);
            model->fit(dataset, features, className, states, weights_);
            models.push_back(std::move(model));
            significanceModels.push_back(1.0);
            n_models++;
        }
        notes.push_back("Used features in initialization: " + std::to_string(featuresUsed.size()) + " of " + std::to_string(features.size()) + " with " + select_features_algorithm);
        delete featureSelector;
        return featuresUsed;
    }
    torch::Tensor BoostAODE::ensemble_predict(torch::Tensor& X, SPODE* model)
    {
        if (initialize_prob_table) {
            initialize_prob_table = false;
            prob_table = model->predict_proba(X) * 1.0;
        } else {
            prob_table += model->predict_proba(X) * 1.0;
        }
        // prob_table doesn't store probabilities but the sum of them
        // to have them we need to divide by the sum of the significances but we
        // don't need them to predict label values
        return prob_table.argmax(1);
    }
    void BoostAODE::trainModel(const torch::Tensor& weights)
    {
        initialize_prob_table = true;
        fitted = true;
        double alpha_t = 0;
        // Algorithm based on the adaboost algorithm for classification
        // as explained in Ensemble methods (Zhi-Hua Zhou, 2012)
        torch::Tensor weights_ = torch::full({ m }, 1.0 / m, torch::kFloat64);
        bool exitCondition = false;
        std::unordered_set<int> featuresUsed;
        if (selectFeatures) {
            featuresUsed = initializeModels();
            auto ypred = predict(X_train);
            std::tie(weights_, alpha_t, exitCondition) = update_weights(y_train, ypred, weights_);
            // Update significance of the models
            for (int i = 0; i < n_models; ++i) {
                significanceModels[i] = alpha_t;
            }
            if (exitCondition) {
                return;
            }
        }
        bool resetMaxModels = false;
        if (maxModels == 0) {
            maxModels = .1 * n > 10 ? .1 * n : n;
            resetMaxModels = true; // Flag to unset maxModels
        }
        // Variables to control the accuracy finish condition
        double priorAccuracy = 0.0;
        double delta = 1.0;
        double convergence_threshold = 1e-4;
        int count = 0; // number of times the accuracy is lower than the convergence_threshold
        // Step 0: Set the finish condition
        // if not repeatSparent a finish condition is run out of features
        // n_models == maxModels
        // epsilon sub t > 0.5 => inverse the weights policy
        // validation error is not decreasing
        bool ascending = order_algorithm == Orders.ASC;
        std::mt19937 g{ 173 };
        while (!exitCondition) {
            // Step 1: Build ranking with mutual information
            auto featureSelection = metrics.SelectKBestWeighted(weights_, ascending, n); // Get all the features sorted
            if (order_algorithm == Orders.RAND) {
                std::shuffle(featureSelection.begin(), featureSelection.end(), g);
            }
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
            std::unique_ptr<Classifier> model;
            model = std::make_unique<SPODE>(feature);
            model->fit(dataset, features, className, states, weights_);
            torch::Tensor ypred;
            if (predict_single) {
                ypred = model->predict(X_train);
            } else {
                ypred = ensemble_predict(X_train, dynamic_cast<SPODE*>(model.get()));
            }
            // Step 3.1: Compute the classifier amout of say
            std::tie(weights_, alpha_t, exitCondition) = update_weights(y_train, ypred, weights_);
            if (exitCondition) {
                break;
            }
            // Step 3.4: Store classifier and its accuracy to weigh its future vote
            featuresUsed.insert(feature);
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
                if (delta < convergence_threshold) {
                    count++;
                }
                priorAccuracy = accuracy;
            }
            exitCondition = n_models >= maxModels && repeatSparent || count > tolerance;
        }
        if (featuresUsed.size() != features.size()) {
            notes.push_back("Used features in train: " + std::to_string(featuresUsed.size()) + " of " + std::to_string(features.size()));
            status = WARNING;
        }
        notes.push_back("Number of models: " + std::to_string(n_models));
        if (resetMaxModels) {
            maxModels = 0;
        }
    }
    std::vector<std::string> BoostAODE::graph(const std::string& title) const
    {
        return Ensemble::graph(title);
    }
}