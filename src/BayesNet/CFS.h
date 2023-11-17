#ifndef CFS_H
#define CFS_H
#include <torch/torch.h>
#include <vector>
#include "FeatureSelect.h"
namespace bayesnet {
    class CFS : public FeatureSelect {
    public:
        // dataset is a n+1xm tensor of integers where dataset[-1] is the y std::vector
        CFS(const torch::Tensor& samples, const std::vector<std::string>& features, const std::string& className, const int maxFeatures, const int classNumStates, const torch::Tensor& weights) :
            FeatureSelect(samples, features, className, maxFeatures, classNumStates, weights)
        {
        }
        virtual ~CFS() {};
        void fit() override;
    private:
        bool computeContinueCondition(const std::vector<int>& featureOrder);
    };
}
#endif