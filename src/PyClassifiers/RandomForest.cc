#include "RandomForest.h"

namespace pywrap {
    RandomForest::RandomForest() : PyClassifier("sklearn.ensemble", "RandomForestClassifier", true)
    {
        validHyperparameters = { "n_estimators", "n_jobs", "random_state" };
    }
} /* namespace pywrap */