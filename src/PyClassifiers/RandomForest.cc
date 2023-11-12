#include "RandomForest.h"

namespace pywrap {
    std::string RandomForest::version()
    {
        return sklearnVersion();
    }
} /* namespace pywrap */