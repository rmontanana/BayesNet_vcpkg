#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H
#include "PyClassifier.h"

namespace pywrap {
    class RandomForest : public PyClassifier {
    public:
        RandomForest();
        ~RandomForest() = default;
    };
} /* namespace pywrap */
#endif /* RANDOMFOREST_H */