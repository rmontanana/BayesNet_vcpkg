#ifndef TANLD_H
#define TANLD_H
#include "TAN.h"
#include "Proposal.h"

namespace bayesnet {
    using namespace std;
    class TANLd : public TAN, public Proposal {
    private:
    public:
        TANLd();
        virtual ~TANLd() = default;
        TANLd& fit(torch::Tensor& X, torch::Tensor& y, vector<string>& features, string className, map<string, vector<int>>& states) override;
        vector<string> graph(const string& name = "TAN") const override;
        Tensor predict(Tensor& X) override;
        static inline string version() { return "0.0.1"; };
        void setHyperparameters(nlohmann::json& hyperparameters) override {};
    };
}
#endif // !TANLD_H