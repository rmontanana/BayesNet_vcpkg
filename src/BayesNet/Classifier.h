#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <torch/torch.h>
#include "BaseClassifier.h"
#include "Network.h"
#include "BayesMetrics.h"
using namespace std;
using namespace torch;

namespace bayesnet {
    class Classifier : public BaseClassifier {
    private:
        bool fitted;
        Classifier& build(vector<string>& features, string className, map<string, vector<int>>& states);
    protected:
        Network model;
        int m, n; // m: number of samples, n: number of features
        Tensor X;
        vector<vector<int>> Xv;
        Tensor y;
        vector<int> yv;
        Tensor dataset;
        Metrics metrics;
        vector<string> features;
        string className;
        map<string, vector<int>> states;
        void checkFitParameters();
        virtual void train() = 0;
    public:
        Classifier(Network model);
        virtual ~Classifier() = default;
        Classifier& fit(vector<vector<int>>& X, vector<int>& y, vector<string>& features, string className, map<string, vector<int>>& states);
        void addNodes();
        int getNumberOfNodes();
        int getNumberOfEdges();
        Tensor predict(Tensor& X);
        vector<int> predict(vector<vector<int>>& X);
        float score(Tensor& X, Tensor& y);
        float score(vector<vector<int>>& X, vector<int>& y);
        vector<string> show();
    };
}
#endif





