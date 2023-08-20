#ifndef SPODE_H
#define SPODE_H
#include "Classifier.h"

namespace bayesnet {
    class SPODE : public Classifier {
    private:
        int root;
    protected:
        void buildModel(const torch::Tensor& weights) override;
    public:
        explicit SPODE(int root);
        virtual ~SPODE() {};
        vector<string> graph(const string& name = "SPODE") const override;
        void setHyperparameters(nlohmann::json& hyperparameters) override {};
    };
}
#endif