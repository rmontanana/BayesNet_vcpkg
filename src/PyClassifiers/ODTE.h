#ifndef ODTE_H
#define ODTE_H
#include "nlohmann/json.hpp"
#include "PyClassifier.h"

namespace pywrap {
    class ODTE : public PyClassifier {
    public:
        ODTE();
        ~ODTE() = default;
        std::string graph();
    };
} /* namespace pywrap */
#endif /* ODTE_H */