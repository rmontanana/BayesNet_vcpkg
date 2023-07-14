#include "AODE.h"

namespace bayesnet {

    AODE::AODE() : Ensemble()
    {
        models = vector<SPODE>();
    }
    void AODE::train()
    {
        for (int i = 0; i < features.size(); ++i) {
            SPODE model = SPODE(i);
            models.push_back(model);
        }
    }
}