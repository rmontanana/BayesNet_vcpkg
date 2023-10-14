#ifndef FCBF_H
#define FCBF_H
#include <torch/torch.h>
#include <vector>
#include "FeatureSelect.h"
using namespace std;
namespace bayesnet {
    class FCBF : public FeatureSelect {
    public:
        // dataset is a n+1xm tensor of integers where dataset[-1] is the y vector
        FCBF(const torch::Tensor& samples, const vector<string>& features, const string& className, const int maxFeatures, const int classNumStates, const torch::Tensor& weights, const double threshold);
        virtual ~FCBF() {};
        void fit() override;
    private:
        double threshold = -1;
    };
}
#endif