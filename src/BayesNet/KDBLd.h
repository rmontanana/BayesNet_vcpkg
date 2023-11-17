#ifndef KDBLD_H
#define KDBLD_H
#include "KDB.h"
#include "Proposal.h"

namespace bayesnet {
    class KDBLd : public KDB, public Proposal {
    private:
    public:
        explicit KDBLd(int k);
        virtual ~KDBLd() = default;
        KDBLd& fit(torch::Tensor& X, torch::Tensor& y, const std::vector<std::string>& features, const std::string& className, map<std::string, std::vector<int>>& states) override;
        std::vector<std::string> graph(const std::string& name = "KDB") const override;
        torch::Tensor predict(torch::Tensor& X) override;
        static inline std::string version() { return "0.0.1"; };
    };
}
#endif // !KDBLD_H