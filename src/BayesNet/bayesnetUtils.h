#ifndef BAYESNET_UTILS_H
#define BAYESNET_UTILS_H
#include <torch/torch.h>
#include <vector>
namespace bayesnet {
    std::vector<int> argsort(std::vector<double>& nums);
    std::vector<std::vector<int>> tensorToVector(torch::Tensor& tensor);
}
#endif //BAYESNET_UTILS_H