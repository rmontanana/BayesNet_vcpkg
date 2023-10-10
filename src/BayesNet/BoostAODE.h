#ifndef BOOSTAODE_H
#define BOOSTAODE_H
#include "Ensemble.h"
#include "SPODE.h"
namespace bayesnet {
    class BoostAODE : public Ensemble {
    public:
        BoostAODE();
        virtual ~BoostAODE() {};
        vector<string> graph(const string& title = "BoostAODE") const override;
        void setHyperparameters(nlohmann::json& hyperparameters) override;
    protected:
        void buildModel(const torch::Tensor& weights) override;
        void trainModel(const torch::Tensor& weights) override;
    private:
        torch::Tensor dataset_;
        torch::Tensor X_train, y_train, X_test, y_test;
        void validationInit();
        void initializeModels();
        // Hyperparameters
        bool repeatSparent = false; // if true, a feature can be selected more than once
        int maxModels = 0;
        bool ascending = false; //Process KBest features ascending or descending order
        bool convergence = false; //if true, stop when the model does not improve
        string cfs = ""; // if not empty, use CFS to select features
    };
}
#endif