#ifndef SPODELD_H
#define SPODELD_H
#include "SPODE.h"
#include "Proposal.h"

namespace bayesnet {
    using namespace std;
    class SPODELd : public SPODE, public Proposal {
    private:
    public:
        explicit SPODELd(int root);
        virtual ~SPODELd() = default;
        SPODELd& fit(torch::Tensor& X, torch::Tensor& y, vector<string>& features, string className, map<string, vector<int>>& states) override;
        vector<string> graph(const string& name = "SPODE") override;
        Tensor predict(Tensor& X) override;
        static inline string version() { return "0.0.1"; };
    };
}
#endif // !SPODELD_H