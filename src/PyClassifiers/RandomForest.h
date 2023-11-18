#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H
#include "PyClassifier.h"

namespace pywrap {
    class RandomForest : public PyClassifier {
    public:
        RandomForest() : PyClassifier("sklearn.ensemble", "RandomForestClassifier", true) {};
        ~RandomForest() = default;
        void setHyperparameters(const nlohmann::json& hyperparameters) override;
    };
} /* namespace pywrap */
#endif /* RANDOMFOREST_H */