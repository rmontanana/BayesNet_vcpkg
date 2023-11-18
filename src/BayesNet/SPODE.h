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
        virtual ~SPODE() = default;
        std::vector<std::string> graph(const std::string& name = "SPODE") const override;
    };
}
#endif