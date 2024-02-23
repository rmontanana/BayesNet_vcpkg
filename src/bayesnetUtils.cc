
#include "bayesnetUtils.h"
namespace bayesnet {
    // Return the indices in descending order
    std::vector<int> argsort(std::vector<double>& nums)
    {
        int n = nums.size();
        std::vector<int> indices(n);
        iota(indices.begin(), indices.end(), 0);
        sort(indices.begin(), indices.end(), [&nums](int i, int j) {return nums[i] > nums[j];});
        return indices;
    }
    template<typename T>
    std::vector<std::vector<T>> tensorToVector(torch::Tensor& dtensor)
    {
        // convert mxn tensor to nxm std::vector
        std::vector<std::vector<T>> result;
        // Iterate over cols
        for (int i = 0; i < dtensor.size(1); ++i) {
            auto col_tensor = dtensor.index({ "...", i });
            auto col = std::vector<T>(col_tensor.data_ptr<T>(), col_tensor.data_ptr<T>() + dtensor.size(0));
            result.push_back(col);
        }
        return result;
    }
    torch::Tensor vectorToTensor(std::vector<std::vector<int>>& vector)
    {
        // convert nxm std::vector to mxn tensor
        long int m = vector[0].size();
        long int n = vector.size();
        auto tensor = torch::zeros({ m, n }, torch::kInt32);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                tensor[i][j] = vector[j][i];
            }
        }
        return tensor;
    }
}