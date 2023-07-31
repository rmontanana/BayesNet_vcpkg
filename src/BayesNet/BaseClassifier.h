#ifndef BASE_H
#define BASE_H
#include <torch/torch.h>
#include <vector>
namespace bayesnet {
    using namespace std;
    class BaseClassifier {
    public:
        virtual BaseClassifier& fit(vector<vector<int>>& X, vector<int>& y, vector<string>& features, string className, map<string, vector<int>>& states) = 0;
        virtual BaseClassifier& fit(torch::Tensor& X, torch::Tensor& y, vector<string>& features, string className, map<string, vector<int>>& states) = 0;
        torch::Tensor virtual predict(torch::Tensor& X) = 0;
        vector<int> virtual predict(vector<vector<int>>& X) = 0;
        float virtual score(vector<vector<int>>& X, vector<int>& y) = 0;
        float virtual score(torch::Tensor& X, torch::Tensor& y) = 0;
        int virtual getNumberOfNodes() = 0;
        int virtual getNumberOfEdges() = 0;
        int virtual getNumberOfStates() = 0;
        vector<string> virtual show() = 0;
        vector<string> virtual graph(const string& title = "") = 0;
        virtual ~BaseClassifier() = default;
        const string inline getVersion() const { return "0.1.0"; };
    };
}
#endif