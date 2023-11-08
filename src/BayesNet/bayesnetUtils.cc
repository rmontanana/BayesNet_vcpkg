
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
    std::vector<std::vector<int>> tensorToVector(torch::Tensor& tensor)
    {
        // convert mxn tensor to nxm std::vector
        std::vector<std::vector<int>> result;
        // Iterate over cols
        for (int i = 0; i < tensor.size(1); ++i) {
            auto col_tensor = tensor.index({ "...", i });
            auto col = std::vector<int>(col_tensor.data_ptr<int>(), col_tensor.data_ptr<int>() + tensor.size(0));
            result.push_back(col);
        }
        return result;
    }
}