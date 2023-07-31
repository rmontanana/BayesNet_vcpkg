#ifndef TANNEW_H
#define TANNEW_H
#include "TAN.h"
#include "CPPFImdlp.h"

namespace bayesnet {
    using namespace std;
    class TANNew : public TAN {
    private:
        mdlp::CPPFImdlp discretizer;
    public:
        TANNew();
        virtual ~TANNew();
        void train() override;
        TANNew& fit(torch::Tensor& X, torch::Tensor& y, vector<string>& features, string className, map<string, vector<int>>& states) override;
        vector<string> graph(const string& name = "TAN") override;
        static inline string version() { return "0.0.1"; };
    };
}

#endif // !TANNEW_H