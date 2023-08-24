#ifndef AODELD_H
#define AODELD_H
#include "Ensemble.h"
#include "Proposal.h"
#include "SPODELd.h"

namespace bayesnet {
    using namespace std;
    class AODELd : public Ensemble, public Proposal {
    protected:
        void trainModel(const torch::Tensor& weights) override;
        void buildModel(const torch::Tensor& weights) override;
    public:
        AODELd();
        AODELd& fit(torch::Tensor& X_, torch::Tensor& y_, const vector<string>& features_, const string& className_, map<string, vector<int>>& states_) override;
        virtual ~AODELd() = default;
        vector<string> graph(const string& name = "AODELd") const override;
        static inline string version() { return "0.0.1"; };
    };
}
#endif // !AODELD_H