#ifndef SPODELD_H
#define SPODELD_H
#include "SPODE.h"
#include "Proposal.h"

namespace bayesnet {
    using namespace std;
    class SPODELd : public SPODE, public Proposal {
    public:
        explicit SPODELd(int root);
        virtual ~SPODELd() = default;
        SPODELd& fit(torch::Tensor& X, torch::Tensor& y, const vector<string>& features, const string& className, map<string, vector<int>>& states) override;
        SPODELd& fit(torch::Tensor& dataset, const vector<string>& features, const string& className, map<string, vector<int>>& states) override;
        vector<string> graph(const string& name = "SPODE") const override;
        Tensor predict(Tensor& X) override;
        void setHyperparameters(nlohmann::json& hyperparameters) override {};
        static inline string version() { return "0.0.1"; };
    };
}
#endif // !SPODELD_H