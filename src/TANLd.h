#ifndef TANLD_H
#define TANLD_H
#include "TAN.h"
#include "Proposal.h"

namespace bayesnet {
    class TANLd : public TAN, public Proposal {
    private:
    public:
        TANLd();
        virtual ~TANLd() = default;
        TANLd& fit(torch::Tensor& X, torch::Tensor& y, const std::vector<std::string>& features, const std::string& className, map<std::string, std::vector<int>>& states) override;
        std::vector<std::string> graph(const std::string& name = "TAN") const override;
        torch::Tensor predict(torch::Tensor& X) override;
        static inline std::string version() { return "0.0.1"; };
    };
}
#endif // !TANLD_H