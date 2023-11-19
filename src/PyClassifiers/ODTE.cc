#include "ODTE.h"

namespace pywrap {
    ODTE::ODTE() : PyClassifier("odte", "Odte")
    {
        validHyperparameters = { "n_jobs", "n_estimators", "random_state" };
    }
    std::string ODTE::graph()
    {
        return callMethodString("graph");
    }
} /* namespace pywrap */