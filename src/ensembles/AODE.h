#ifndef AODE_H
#define AODE_H
#include "Ensemble.h"
#include "SPODE.h"
namespace bayesnet {
    class AODE : public Ensemble {
    public:
        AODE(bool predict_voting = true);
        virtual ~AODE() {};
        void setHyperparameters(const nlohmann::json& hyperparameters) override;
        std::vector<std::string> graph(const std::string& title = "AODE") const override;
    protected:
        void buildModel(const torch::Tensor& weights) override;
    };
}
#endif