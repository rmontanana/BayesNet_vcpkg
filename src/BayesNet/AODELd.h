#ifndef AODELD_H
#define AODELD_H
#include "Ensemble.h"
#include "Proposal.h"
#include "SPODELd.h"

namespace bayesnet {
    using namespace std;
    class AODELd : public Ensemble, public Proposal {
    protected:
        void trainModel() override;
        void buildModel() override;
    public:
        AODELd();
        AODELd& fit(torch::Tensor& X_, torch::Tensor& y_, vector<string>& features_, string className_, map<string, vector<int>>& states_) override;
        virtual ~AODELd() = default;
        vector<string> graph(const string& name = "AODE") const override;
        static inline string version() { return "0.0.1"; };
    };
}
#endif // !AODELD_H