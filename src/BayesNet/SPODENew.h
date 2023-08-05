#ifndef SPODENEW_H
#define SPODENEW_H
#include "SPODE.h"
#include "Proposal.h"

namespace bayesnet {
    using namespace std;
    class SPODENew : public SPODE, public Proposal {
    private:
    public:
        explicit SPODENew(int root);
        virtual ~SPODENew() = default;
        SPODENew& fit(torch::Tensor& X, torch::Tensor& y, vector<string>& features, string className, map<string, vector<int>>& states) override;
        vector<string> graph(const string& name = "SPODE") override;
        Tensor predict(Tensor& X) override;
        static inline string version() { return "0.0.1"; };
    };
}
#endif // !SPODENew_H