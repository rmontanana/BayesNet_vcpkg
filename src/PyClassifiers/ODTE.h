#ifndef ODTE_H
#define ODTE_H
#include "nlohmann/json.hpp"
#include "PyClassifier.h"

namespace pywrap {
    class ODTE : public PyClassifier {
    public:
        ODTE() : PyClassifier("odte", "Odte") {};
        ~ODTE() = default;
        std::string graph();
        void setHyperparameters(nlohmann::json& hyperparameters) override;
    };
} /* namespace pywrap */
#endif /* ODTE_H */