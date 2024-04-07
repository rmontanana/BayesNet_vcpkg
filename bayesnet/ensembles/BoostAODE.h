#ifndef BOOSTAODE_H
#define BOOSTAODE_H
#include <map>
#include "bayesnet/classifiers/SPODE.h"
#include "bayesnet/feature_selection/FeatureSelect.h"
#include "Ensemble.h"
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
    class BoostAODE : public Ensemble {
    public:
        BoostAODE(bool predict_voting = false);
        virtual ~BoostAODE() = default;
        std::vector<std::string> graph(const std::string& title = "BoostAODE") const override;
        void setHyperparameters(const nlohmann::json& hyperparameters_) override;
    protected:
        void buildModel(const torch::Tensor& weights) override;
        void trainModel(const torch::Tensor& weights) override;
    private:
        std::vector<int> initializeModels();
        torch::Tensor X_train, y_train, X_test, y_test;
        // Hyperparameters
        bool bisection = false; // if true, use bisection stratety to add k models at once to the ensemble
        int maxTolerance = 1;
        std::string order_algorithm; // order to process the KBest features asc, desc, rand
        bool convergence = false; //if true, stop when the model does not improve
        bool selectFeatures = false; // if true, use feature selection
        std::string select_features_algorithm = Orders.DESC; // Selected feature selection algorithm
        FeatureSelect* featureSelector = nullptr;
        double threshold = -1;
    };
}
#endif