#ifndef BAYESNET_UTILS_H
#define BAYESNET_UTILS_H
#include <torch/torch.h>
#include <vector>
namespace bayesnet {
    std::vector<int> argsort(std::vector<double>& nums);
    std::vector<std::vector<int>> tensorToVector(torch::Tensor& dtensor);
    std::vector<std::vector<double>> tensorToVectorDouble(torch::Tensor& dtensor);
    torch::Tensor vectorToTensor(std::vector<std::vector<int>>& vector, bool transpose = true);
}
#endif //BAYESNET_UTILS_H