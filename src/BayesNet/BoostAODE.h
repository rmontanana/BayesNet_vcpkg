#ifndef BOOSTAODE_H
#define BOOSTAODE_H
#include "Ensemble.h"
#include <map>
#include "SPODE.h"
#include "FeatureSelect.h"
namespace bayesnet {
    class BoostAODE : public Ensemble {
    public:
        BoostAODE();
        virtual ~BoostAODE() = default;
        std::vector<std::string> graph(const std::string& title = "BoostAODE") const override;
        void setHyperparameters(const nlohmann::json& hyperparameters) override;
    protected:
        void buildModel(const torch::Tensor& weights) override;
        void trainModel(const torch::Tensor& weights) override;
    private:
        torch::Tensor dataset_;
        torch::Tensor X_train, y_train, X_test, y_test;
        std::unordered_set<int> initializeModels();
        // Hyperparameters
        bool repeatSparent = false; // if true, a feature can be selected more than once
        int maxModels = 0;
        bool ascending = false; //Process KBest features ascending or descending order
        bool convergence = false; //if true, stop when the model does not improve
        bool selectFeatures = false; // if true, use feature selection
        std::string algorithm = ""; // Selected feature selection algorithm
        FeatureSelect* featureSelector = nullptr;
        double threshold = -1;
    };
}
#endif