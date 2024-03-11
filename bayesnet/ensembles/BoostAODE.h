#ifndef BOOSTAODE_H
#define BOOSTAODE_H
#include <map>
#include "bayesnet/classifiers/SPODE.h"
#include "bayesnet/feature_selection/FeatureSelect.h"
#include "Ensemble.h"
namespace bayesnet {
    class BoostAODE : public Ensemble {
    public:
        BoostAODE(bool predict_voting = false);
        virtual ~BoostAODE() = default;
        std::vector<std::string> graph(const std::string& title = "BoostAODE") const override;
        void setHyperparameters(const nlohmann::json& hyperparameters) override;
    protected:
        void buildModel(const torch::Tensor& weights) override;
        void trainModel(const torch::Tensor& weights) override;
    private:
        std::unordered_set<int> initializeModels();
        torch::Tensor ensemble_predict(torch::Tensor& X, SPODE* model);
        torch::Tensor dataset_;
        torch::Tensor X_train, y_train, X_test, y_test;
        // Hyperparameters
        bool repeatSparent = false; // if true, a feature can be selected more than once
        int maxModels = 0;
        int tolerance = 0;
        bool predict_single = true; // wether the last model is used to predict in training or the whole ensemble
        std::string order_algorithm; // order to process the KBest features asc, desc, rand
        bool convergence = false; //if true, stop when the model does not improve
        bool selectFeatures = false; // if true, use feature selection
        std::string select_features_algorithm = "desc"; // Selected feature selection algorithm
        bool initialize_prob_table; // if true, initialize the prob_table with the first model (used in train)
        torch::Tensor prob_table; // Table of probabilities for ensemble predicting if predict_single is false
        FeatureSelect* featureSelector = nullptr;
        double threshold = -1;
    };
}
#endif