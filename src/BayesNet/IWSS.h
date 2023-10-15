#ifndef IWSS_H
#define IWSS_H
#include <torch/torch.h>
#include <vector>
#include "FeatureSelect.h"
using namespace std;
namespace bayesnet {
    class IWSS : public FeatureSelect {
    public:
        // dataset is a n+1xm tensor of integers where dataset[-1] is the y vector
        IWSS(const torch::Tensor& samples, const vector<string>& features, const string& className, const int maxFeatures, const int classNumStates, const torch::Tensor& weights, const double threshold);
        virtual ~IWSS() {};
        void fit() override;
    private:
        double threshold = -1;
    };
}
#endif