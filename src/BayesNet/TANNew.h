#ifndef TANNEW_H
#define TANNEW_H
#include "TAN.h"
#include "CPPFImdlp.h"

namespace bayesnet {
    using namespace std;
    class TANNew : public TAN {
    private:
        map<string, mdlp::CPPFImdlp*> discretizers;
        int n_features;
        torch::Tensor Xf; // X continuous
    public:
        TANNew();
        virtual ~TANNew();
        void train() override;
        TANNew& fit(torch::Tensor& X, torch::Tensor& y, vector<string>& features, string className, map<string, vector<int>>& states) override;
        vector<string> graph(const string& name = "TAN") override;
        Tensor predict(Tensor& X) override;
        static inline string version() { return "0.0.1"; };
    };
}
#endif // !TANNEW_H