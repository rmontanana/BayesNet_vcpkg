#ifndef BAYESNET_UTILS_H
#define BAYESNET_UTILS_H
#include <torch/torch.h>
#include <vector>
namespace bayesnet {
    std::vector<int> argsort(std::vector<double>& nums);
    template<typename T>
    std::vector<std::vector<T>> tensorToVector(torch::Tensor& dtensor);
    torch::Tensor vectorToTensor(std::vector<std::vector<int>>& vector);
}
#endif //BAYESNET_UTILS_H