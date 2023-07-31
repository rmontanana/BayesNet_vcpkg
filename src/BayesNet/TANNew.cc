#include "TANNew.h"

namespace bayesnet {
    using namespace std;
    TANNew::TANNew() : TAN(), discretizer{ mdlp::CPPFImdlp() } {}
    TANNew::~TANNew() {}
    TANNew& TANNew::fit(torch::Tensor& X, torch::Tensor& y, vector<string>& features, string className, map<string, vector<int>>& states)
    {
        /*
        Hay que discretizar los datos de entrada y luego en predict discretizar también con el mmismo modelo, hacer un transform solamente.
        */
        TAN::fit(X, y, features, className, states);
        return *this;
    }
    void TANNew::train()
    {
        TAN::train();
    }
    vector<string> TANNew::graph(const string& name)
    {
        return TAN::graph(name);
    }
}