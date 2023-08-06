#ifndef AODELD_H
#define AODELD_H
#include "Ensemble.h"
#include "Proposal.h"
#include "SPODELd.h"

namespace bayesnet {
    using namespace std;
    class AODELd : public Ensemble, public Proposal {
    public:
        AODELd();
        virtual ~AODELd() = default;
        AODELd& fit(torch::Tensor& X, torch::Tensor& y, vector<string>& features, string className, map<string, vector<int>>& states) override;
        vector<string> graph(const string& name = "AODE") override;
        Tensor predict(Tensor& X) override;
        void train() override;
        static inline string version() { return "0.0.1"; };
    };
}
#endif // !AODELD_H