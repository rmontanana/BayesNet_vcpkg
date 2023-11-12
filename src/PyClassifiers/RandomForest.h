#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H
#include "PyClassifier.h"

namespace pywrap {
    class RandomForest : public PyClassifier {
    public:
        RandomForest() : PyClassifier("sklearn.ensemble", "RandomForestClassifier") {};
        ~RandomForest() = default;
        std::string version();
    };
} /* namespace pywrap */
#endif /* RANDOMFOREST_H */