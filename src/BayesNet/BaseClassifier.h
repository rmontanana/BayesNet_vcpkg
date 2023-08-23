#ifndef BASE_H
#define BASE_H
#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <vector>
namespace bayesnet {
    using namespace std;
    class BaseClassifier {
    protected:
        virtual void trainModel(const torch::Tensor& weights) = 0;
    public:
        // X is nxm vector, y is nx1 vector
        virtual BaseClassifier& fit(vector<vector<int>>& X, vector<int>& y, const vector<string>& features, const string& className, map<string, vector<int>>& states) = 0;
        // X is nxm tensor, y is nx1 tensor
        virtual BaseClassifier& fit(torch::Tensor& X, torch::Tensor& y, const vector<string>& features, const string& className, map<string, vector<int>>& states) = 0;
        virtual BaseClassifier& fit(torch::Tensor& dataset, const vector<string>& features, const string& className, map<string, vector<int>>& states) = 0;
        virtual BaseClassifier& fit(torch::Tensor& dataset, const vector<string>& features, const string& className, map<string, vector<int>>& states, const torch::Tensor& weights) = 0;
        virtual ~BaseClassifier() = default;
        torch::Tensor virtual predict(torch::Tensor& X) = 0;
        vector<int> virtual predict(vector<vector<int>>& X) = 0;
        float virtual score(vector<vector<int>>& X, vector<int>& y) = 0;
        float virtual score(torch::Tensor& X, torch::Tensor& y) = 0;
        int virtual getNumberOfNodes()const = 0;
        int virtual getNumberOfEdges()const = 0;
        int virtual getNumberOfStates() const = 0;
        vector<string> virtual show() const = 0;
        vector<string> virtual graph(const string& title = "") const = 0;
        const string inline getVersion() const { return "0.1.0"; };
        vector<string> virtual topological_order() = 0;
        void virtual dump_cpt()const = 0;
        virtual void setHyperparameters(nlohmann::json& hyperparameters) = 0;
    };
}
#endif