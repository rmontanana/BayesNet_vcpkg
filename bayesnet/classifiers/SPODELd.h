#ifndef SPODELD_H
#define SPODELD_H
#include "SPODE.h"
#include "Proposal.h"

namespace bayesnet {
    class SPODELd : public SPODE, public Proposal {
    public:
        explicit SPODELd(int root);
        virtual ~SPODELd() = default;
        SPODELd& fit(torch::Tensor& X, torch::Tensor& y, const std::vector<std::string>& features, const std::string& className, map<std::string, std::vector<int>>& states) override;
        SPODELd& fit(torch::Tensor& dataset, const std::vector<std::string>& features, const std::string& className, map<std::string, std::vector<int>>& states) override;
        SPODELd& commonFit(const std::vector<std::string>& features, const std::string& className, map<std::string, std::vector<int>>& states);
        std::vector<std::string> graph(const std::string& name = "SPODE") const override;
        torch::Tensor predict(torch::Tensor& X) override;
        static inline std::string version() { return "0.0.1"; };
    };
}
#endif // !SPODELD_H