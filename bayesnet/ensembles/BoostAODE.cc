#include <set>
#include <functional>
#include <limits.h>
#include <tuple>
#include <folding.hpp>
#include "bayesnet/feature_selection/CFS.h"
#include "bayesnet/feature_selection/FCBF.h"
#include "bayesnet/feature_selection/IWSS.h"
#include "BoostAODE.h"

#include "bayesnet/utils/loguru.cpp"

namespace bayesnet {

    BoostAODE::BoostAODE(bool predict_voting) : Ensemble(predict_voting)
    {
        validHyperparameters = {
            "maxModels", "bisection", "order", "convergence", "threshold",
            "select_features", "maxTolerance", "predict_voting"
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
            // Build dataset with train data
            buildDataset(y_train);
            metrics = Metrics(dataset, features, className, n_classes);
        } else {
            // Use all data to train
            X_train = dataset.index({ torch::indexing::Slice(0, dataset.size(0) - 1), "..." });
            y_train = y_;
        }
    }
    void BoostAODE::setHyperparameters(const nlohmann::json& hyperparameters_)
    {
        auto hyperparameters = hyperparameters_;
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
        if (hyperparameters.contains("bisection")) {
            bisection = hyperparameters["bisection"];
            hyperparameters.erase("bisection");
        }
        if (hyperparameters.contains("threshold")) {
            threshold = hyperparameters["threshold"];
            hyperparameters.erase("threshold");
        }
        if (hyperparameters.contains("maxTolerance")) {
            maxTolerance = hyperparameters["maxTolerance"];
            if (maxTolerance < 1 || maxTolerance > 4)
                throw std::invalid_argument("Invalid maxTolerance value, must be greater in [1, 4]");
            hyperparameters.erase("maxTolerance");
        }
        if (hyperparameters.contains("predict_voting")) {
            predict_voting = hyperparameters["predict_voting"];
            hyperparameters.erase("predict_voting");
        }
        if (hyperparameters.contains("select_features")) {
            auto selectedAlgorithm = hyperparameters["select_features"];
            std::vector<std::string> algos = { SelectFeatures.IWSS, SelectFeatures.CFS, SelectFeatures.FCBF };
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
    std::vector<int> BoostAODE::initializeModels()
    {
        std::vector<int> featuresUsed;
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
            featuresUsed.push_back(feature);
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
    void BoostAODE::trainModel(const torch::Tensor& weights)
    {
        //
        // Logging setup
        //
        loguru::set_thread_name("BoostAODE");
        loguru::g_stderr_verbosity = loguru::Verbosity_OFF;;
        loguru::add_file("boostAODE.log", loguru::Truncate, loguru::Verbosity_MAX);
        // Algorithm based on the adaboost algorithm for classification
        // as explained in Ensemble methods (Zhi-Hua Zhou, 2012)
        fitted = true;
        double alpha_t = 0;
        torch::Tensor weights_ = torch::full({ m }, 1.0 / m, torch::kFloat64);
        bool finished = false;
        std::vector<int> featuresUsed;
        if (selectFeatures) {
            featuresUsed = initializeModels();
            auto ypred = predict(X_train);
            std::tie(weights_, alpha_t, finished) = update_weights(y_train, ypred, weights_);
            // Update significance of the models
            for (int i = 0; i < n_models; ++i) {
                significanceModels[i] = alpha_t;
            }
            if (finished) {
                return;
            }
            LOG_F(INFO, "Initial models: %d", n_models);
            LOG_F(INFO, "Significances: ");
            for (int i = 0; i < n_models; ++i) {
                LOG_F(INFO, "i=%d significance=%f", i, significanceModels[i]);
            }
        }
        int numItemsPack = 0; // The counter of the models inserted in the current pack
        // Variables to control the accuracy finish condition
        double priorAccuracy = 0.0;
        double improvement = 1.0;
        double convergence_threshold = 1e-4;
        int tolerance = 0; // number of times the accuracy is lower than the convergence_threshold
        // Step 0: Set the finish condition
        // epsilon sub t > 0.5 => inverse the weights policy
        // validation error is not decreasing
        // run out of features
        bool ascending = order_algorithm == Orders.ASC;
        std::mt19937 g{ 173 };
        while (!finished) {
            // Step 1: Build ranking with mutual information
            auto featureSelection = metrics.SelectKBestWeighted(weights_, ascending, n); // Get all the features sorted
            VLOG_SCOPE_F(1, "featureSelection.size: %zu featuresUsed.size: %zu", featureSelection.size(), featuresUsed.size());
            if (order_algorithm == Orders.RAND) {
                std::shuffle(featureSelection.begin(), featureSelection.end(), g);
            }
            // Remove used features
            featureSelection.erase(remove_if(begin(featureSelection), end(featureSelection), [&](auto x)
                { return std::find(begin(featuresUsed), end(featuresUsed), x) != end(featuresUsed);}),
                end(featureSelection)
            );
            int k = pow(2, tolerance);
            int counter = 0; // The model counter of the current pack
            VLOG_SCOPE_F(1, "k=%d featureSelection.size: %zu", k, featureSelection.size());
            while (counter++ < k && featureSelection.size() > 0) {
                VLOG_SCOPE_F(2, "counter: %d numItemsPack: %d", counter, numItemsPack);
                auto feature = featureSelection[0];
                featureSelection.erase(featureSelection.begin());
                std::unique_ptr<Classifier> model;
                model = std::make_unique<SPODE>(feature);
                model->fit(dataset, features, className, states, weights_);
                torch::Tensor ypred;
                ypred = model->predict(X_train);
                // Step 3.1: Compute the classifier amout of say
                std::tie(weights_, alpha_t, finished) = update_weights(y_train, ypred, weights_);
                if (finished) {
                    VLOG_SCOPE_F(2, "** epsilon_t > 0.5 **");
                    break;
                }
                // Step 3.4: Store classifier and its accuracy to weigh its future vote
                numItemsPack++;
                featuresUsed.push_back(feature);
                models.push_back(std::move(model));
                significanceModels.push_back(alpha_t);
                n_models++;
                VLOG_SCOPE_F(2, "numItemsPack: %d n_models: %d featuresUsed: %zu", numItemsPack, n_models, featuresUsed.size());
            }
            if (convergence && !finished) {
                auto y_val_predict = predict(X_test);
                double accuracy = (y_val_predict == y_test).sum().item<double>() / (double)y_test.size(0);
                if (priorAccuracy == 0) {
                    priorAccuracy = accuracy;
                    VLOG_SCOPE_F(3, "First accuracy: %f", priorAccuracy);
                } else {
                    improvement = accuracy - priorAccuracy;
                }
                if (improvement < convergence_threshold) {
                    VLOG_SCOPE_F(3, "(improvement<threshold) tolerance: %d numItemsPack: %d improvement: %f prior: %f current: %f", tolerance, numItemsPack, improvement, priorAccuracy, accuracy);
                    tolerance++;
                } else {
                    VLOG_SCOPE_F(3, "*(improvement>=threshold) Reset. tolerance: %d numItemsPack: %d improvement: %f prior: %f current: %f", tolerance, numItemsPack, improvement, priorAccuracy, accuracy);
                    tolerance = 0; // Reset the counter if the model performs better
                    numItemsPack = 0;
                }
                // Keep the best accuracy until now as the prior accuracy
                priorAccuracy = std::max(accuracy, priorAccuracy);
                // priorAccuracy = accuracy;
            }
            VLOG_SCOPE_F(1, "tolerance: %d featuresUsed.size: %zu features.size: %zu", tolerance, featuresUsed.size(), features.size());
            finished = finished || tolerance > maxTolerance || featuresUsed.size() == features.size();
        }
        if (tolerance > maxTolerance) {
            if (numItemsPack < n_models) {
                notes.push_back("Convergence threshold reached & " + std::to_string(numItemsPack) + " models eliminated");
                VLOG_SCOPE_F(4, "Convergence threshold reached & %d models eliminated of %d", numItemsPack, n_models);
                for (int i = 0; i < numItemsPack; ++i) {
                    significanceModels.pop_back();
                    models.pop_back();
                    n_models--;
                }
            } else {
                VLOG_SCOPE_F(4, "Convergence threshold reached & 0 models eliminated n_models=%d numItemsPack=%d", n_models, numItemsPack);
                notes.push_back("Convergence threshold reached & 0 models eliminated");
            }
        }
        if (featuresUsed.size() != features.size()) {
            notes.push_back("Used features in train: " + std::to_string(featuresUsed.size()) + " of " + std::to_string(features.size()));
            status = WARNING;
        }
        notes.push_back("Number of models: " + std::to_string(n_models));
    }
    std::vector<std::string> BoostAODE::graph(const std::string& title) const
    {
        return Ensemble::graph(title);
    }
}