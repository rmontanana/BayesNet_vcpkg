#ifndef AODE_H
#define AODE_H
#include "Ensemble.h"
#include "SPODE.h"
namespace bayesnet {
    class AODE : public Ensemble {
    protected:
        void train() override;
    public:
        AODE();
    };
}
#endif