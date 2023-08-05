#ifndef KDBLD_H
#define KDBLD_H
#include "KDB.h"
#include "Proposal.h"

namespace bayesnet {
    using namespace std;
    class KDBLd : public KDB, public Proposal {
    private:
    public:
        explicit KDBLd(int k);
        virtual ~KDBLd() = default;
        KDBLd& fit(torch::Tensor& X, torch::Tensor& y, vector<string>& features, string className, map<string, vector<int>>& states) override;
        vector<string> graph(const string& name = "KDB") override;
        Tensor predict(Tensor& X) override;
        static inline string version() { return "0.0.1"; };
    };
}
#endif // !KDBLD_H