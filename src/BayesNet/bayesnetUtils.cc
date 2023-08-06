
#include "bayesnetUtils.h"
namespace bayesnet {
    using namespace std;
    using namespace torch;
    // Return the indices in descending order
    vector<int> argsort(vector<float>& nums)
    {
        int n = nums.size();
        vector<int> indices(n);
        iota(indices.begin(), indices.end(), 0);
        sort(indices.begin(), indices.end(), [&nums](int i, int j) {return nums[i] > nums[j];});
        return indices;
    }
    vector<vector<int>> tensorToVector(Tensor& tensor)
    {
        // convert mxn tensor to nxm vector
        vector<vector<int>> result;
        // Iterate over cols
        for (int i = 0; i < tensor.size(1); ++i) {
            auto col_tensor = tensor.index({ "...", i });
            auto col = vector<int>(col_tensor.data_ptr<int>(), col_tensor.data_ptr<int>() + tensor.size(0));
            result.push_back(col);
        }
        return result;
    }
}