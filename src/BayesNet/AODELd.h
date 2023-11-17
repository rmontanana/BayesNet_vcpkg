#ifndef AODELD_H
#define AODELD_H
#include "Ensemble.h"
#include "Proposal.h"
#include "SPODELd.h"

namespace bayesnet {
    class AODELd : public Ensemble, public Proposal {
    protected:
        void trainModel(const torch::Tensor& weights) override;
        void buildModel(const torch::Tensor& weights) override;
    public:
        AODELd();
        AODELd& fit(torch::Tensor& X_, torch::Tensor& y_, const std::vector<std::string>& features_, const std::string& className_, map<std::string, std::vector<int>>& states_) override;
        virtual ~AODELd() = default;
        std::vector<std::string> graph(const std::string& name = "AODELd") const override;
        static inline std::string version() { return "0.0.1"; };
    };
}
#endif // !AODELD_H