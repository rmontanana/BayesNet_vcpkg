#ifndef BOOSTAODE_H
#define BOOSTAODE_H
#include "Ensemble.h"
#include "SPODE.h"
namespace bayesnet {
    class BoostAODE : public Ensemble {
    protected:
        void buildModel(const torch::Tensor& weights) override;
    public:
        BoostAODE();
        virtual ~BoostAODE() {};
        vector<string> graph(const string& title = "BoostAODE") const override;
    };
}
#endif