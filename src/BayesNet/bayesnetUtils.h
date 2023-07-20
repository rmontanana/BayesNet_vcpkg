#ifndef BAYESNET_UTILS_H
#define BAYESNET_UTILS_H
#include <torch/torch.h>
#include <vector>
namespace bayesnet {
    using namespace std;
    using namespace torch;
    vector<int> argsort(vector<float>& nums);
    vector<vector<int>> tensorToVector(const Tensor& tensor);
}
#endif //BAYESNET_UTILS_H