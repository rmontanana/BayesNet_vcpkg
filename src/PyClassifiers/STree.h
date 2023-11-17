#ifndef STREE_H
#define STREE_H
#include "nlohmann/json.hpp"
#include "PyClassifier.h"

namespace pywrap {
    class STree : public PyClassifier {
    public:
        STree() : PyClassifier("stree", "Stree") {};
        ~STree() = default;
        std::string graph();
        void setHyperparameters(nlohmann::json& hyperparameters) override;
    };
} /* namespace pywrap */
#endif /* STREE_H */