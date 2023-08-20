#ifndef AODE_H
#define AODE_H
#include "Ensemble.h"
#include "SPODE.h"
namespace bayesnet {
    class AODE : public Ensemble {
    protected:
        void buildModel(const torch::Tensor& weights) override;
    public:
        AODE();
        virtual ~AODE() {};
        vector<string> graph(const string& title = "AODE") const override;
        void setHyperparameters(nlohmann::json& hyperparameters) override {};
    };
}
#endif