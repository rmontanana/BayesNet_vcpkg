#ifndef KDBNEW_H
#define KDBNEW_H
#include "KDB.h"
#include "Proposal.h"

namespace bayesnet {
    using namespace std;
    class KDBNew : public KDB, public Proposal {
    private:
    public:
        explicit KDBNew(int k);
        virtual ~KDBNew() = default;
        KDBNew& fit(torch::Tensor& X, torch::Tensor& y, vector<string>& features, string className, map<string, vector<int>>& states) override;
        vector<string> graph(const string& name = "KDB") override;
        Tensor predict(Tensor& X) override;
        static inline string version() { return "0.0.1"; };
    };
}
#endif // !KDBNew_H