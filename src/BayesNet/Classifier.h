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
        void buildDataset(torch::Tensor& y);
        Classifier& build(vector<string>& features, string className, map<string, vector<int>>& states);
    protected:
        bool fitted;
        int m, n; // m: number of samples, n: number of features
        Network model;
        Metrics metrics;
        vector<string> features;
        string className;
        map<string, vector<int>> states;
        Tensor dataset; // (n+1)xm tensor
        Tensor weights;
        void checkFitParameters();
        virtual void buildModel() = 0;
        void trainModel() override;
    public:
        Classifier(Network model);
        virtual ~Classifier() = default;
        Classifier& fit(vector<vector<int>>& X, vector<int>& y, vector<string>& features, string className, map<string, vector<int>>& states) override;
        Classifier& fit(torch::Tensor& X, torch::Tensor& y, vector<string>& features, string className, map<string, vector<int>>& states) override;
        Classifier& fit(torch::Tensor& dataset, vector<string>& features, string className, map<string, vector<int>>& states) override;
        void addNodes();
        int getNumberOfNodes() const override;
        int getNumberOfEdges() const override;
        int getNumberOfStates() const override;
        Tensor predict(Tensor& X) override;
        vector<int> predict(vector<vector<int>>& X) override;
        float score(Tensor& X, Tensor& y) override;
        float score(vector<vector<int>>& X, vector<int>& y) override;
        vector<string> show() const override;
        vector<string> topological_order()  override;
        void dump_cpt() const override;
    };
}
#endif





