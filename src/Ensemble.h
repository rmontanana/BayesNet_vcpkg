#ifndef ENSEMBLE_H
#define ENSEMBLE_H
#include <torch/torch.h>
#include "BaseClassifier.h"
#include "Metrics.hpp"
#include "utils.h"
using namespace std;
using namespace torch;

namespace bayesnet {
    class Ensemble {
    private:
        long n_models;
        Ensemble& build(vector<string>& features, string className, map<string, vector<int>>& states);
    protected:
        vector<BaseClassifier> models;
        int m, n; // m: number of samples, n: number of features
        Tensor X;
        Tensor y;
        Tensor dataset;
        Metrics metrics;
        vector<string> features;
        string className;
        map<string, vector<int>> states;
        void virtual train() = 0;
    public:
        Ensemble();
        virtual ~Ensemble() = default;
        Ensemble& fit(Tensor& X, Tensor& y, vector<string>& features, string className, map<string, vector<int>>& states);
        Ensemble& fit(vector<vector<int>>& X, vector<int>& y, vector<string>& features, string className, map<string, vector<int>>& states);
        Tensor predict(Tensor& X);
        float score(Tensor& X, Tensor& y);
        vector<string> show();
    };
}
#endif
