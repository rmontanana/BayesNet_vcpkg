#ifndef BASE_H
#define BASE_H
#include <torch/torch.h>
#include <vector>
namespace bayesnet {
    using namespace std;
    class BaseClassifier {
    public:
        virtual BaseClassifier& fit(vector<vector<int>>& X, vector<int>& y, vector<string>& features, string className, map<string, vector<int>>& states) = 0;
        vector<int> virtual predict(vector<vector<int>>& X) = 0;
        float virtual score(vector<vector<int>>& X, vector<int>& y) = 0;
        vector<string> virtual show() = 0;
        vector<string> virtual graph(string title = "") = 0;
        virtual ~BaseClassifier() = default;
    };
}
#endif