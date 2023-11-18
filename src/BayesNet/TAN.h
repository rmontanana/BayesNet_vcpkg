#ifndef TAN_H
#define TAN_H
#include "Classifier.h"
namespace bayesnet {
    class TAN : public Classifier {
    private:
    protected:
        void buildModel(const torch::Tensor& weights) override;
    public:
        TAN();
        virtual ~TAN() = default;
        std::vector<std::string> graph(const std::string& name = "TAN") const override;
    };
}
#endif