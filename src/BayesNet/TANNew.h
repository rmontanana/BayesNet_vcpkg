#ifndef TANNEW_H
#define TANNEW_H
#include "TAN.h"
#include "Proposal.h"

namespace bayesnet {
    using namespace std;
    class TANNew : public TAN, public Proposal {
    private:
    public:
        TANNew();
        virtual ~TANNew();
        TANNew& fit(torch::Tensor& X, torch::Tensor& y, vector<string>& features, string className, map<string, vector<int>>& states) override;
        vector<string> graph(const string& name = "TAN") override;
        Tensor predict(Tensor& X) override;
        static inline string version() { return "0.0.1"; };
    };
}
#endif // !TANNEW_H