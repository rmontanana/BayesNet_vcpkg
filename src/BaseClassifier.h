#ifndef CLASSIFIERS_H
#define CLASSIFIERS_H
#include <torch/torch.h>
#include "Network.h"
#include "Metrics.hpp"
using namespace std;
using namespace torch;

namespace bayesnet {
    class BaseClassifier {
    private:
        BaseClassifier& build(vector<string>& features, string className, map<string, vector<int>>& states);
    protected:
        Network model;
        int m, n; // m: number of samples, n: number of features
        Tensor X;
        Tensor y;
        Tensor dataset;
        Metrics metrics;
        vector<string> features;
        string className;
        map<string, vector<int>> states;
        void checkFitParameters();
        virtual void train() = 0;
    public:
        BaseClassifier(Network model);
        BaseClassifier& fit(Tensor& X, Tensor& y, vector<string>& features, string className, map<string, vector<int>>& states);
        BaseClassifier& fit(vector<vector<int>>& X, vector<int>& y, vector<string>& features, string className, map<string, vector<int>>& states);
        void addNodes();
        Tensor predict(Tensor& X);
        float score(Tensor& X, Tensor& y);
        vector<string> show();
        vector<int> argsort(vector<float>& nums);
    };
}
#endif





