#ifndef CFS_H
#define CFS_H
#include <torch/torch.h>
#include <vector>
#include "FeatureSelect.h"
using namespace std;
namespace bayesnet {
    class CFS : public FeatureSelect {
    public:
        // dataset is a n+1xm tensor of integers where dataset[-1] is the y vector
        CFS(const torch::Tensor& samples, const vector<string>& features, const string& className, const int maxFeatures, const int classNumStates, const torch::Tensor& weights) :
            FeatureSelect(samples, features, className, maxFeatures, classNumStates, weights)
        {
        }
        virtual ~CFS() {};
        void fit() override;
    private:
        bool computeContinueCondition(const vector<int>& featureOrder);
    };
}
#endif