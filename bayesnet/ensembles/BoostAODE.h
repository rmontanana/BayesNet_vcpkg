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
        torch::Tensor dataset_;
        torch::Tensor X_train, y_train, X_test, y_test;
        // Hyperparameters
        bool bisection = false; // if true, use bisection stratety to add k models at once to the ensemble
        int maxTolerance = 1;
        std::string order_algorithm; // order to process the KBest features asc, desc, rand
        bool convergence = false; //if true, stop when the model does not improve
        bool selectFeatures = false; // if true, use feature selection
        std::string select_features_algorithm = "desc"; // Selected feature selection algorithm
        FeatureSelect* featureSelector = nullptr;
        double threshold = -1;
    };
}
#endif