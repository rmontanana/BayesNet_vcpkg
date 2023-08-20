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
        bool repeatSparent;
    };
}
#endif