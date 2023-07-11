// #include <torch/torch.h>

// int main()
// {
//     torch::Tensor t = torch::rand({ 5, 5 });

//     // Print original tensor
//     std::cout << t << std::endl;

//     // New value
//     torch::Tensor new_val = torch::tensor(10.0f);

//     // Indices for the cell you want to update
//     auto index_i = torch::tensor({ 2 });
//     auto index_j = torch::tensor({ 3 });

//     // Update cell
//     t.index_put_({ index_i, index_j }, new_val);

//     // Print updated tensor
//     std::cout << t << std::endl;
// }
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>
using namespace std;
double entropy(torch::Tensor feature)
{
    torch::Tensor counts = feature.bincount();
    int totalWeight = counts.sum().item<int>();
    torch::Tensor probs = counts.to(torch::kFloat) / totalWeight;
    torch::Tensor logProbs = torch::log2(probs);
    torch::Tensor entropy = -probs * logProbs;
    return entropy.sum().item<double>();
}
// H(Y|X) = sum_{x in X} p(x) H(Y|X=x)
double conditionalEntropy(torch::Tensor firstFeature, torch::Tensor secondFeature)
{
    int numSamples = firstFeature.sizes()[0];
    torch::Tensor featureCounts = secondFeature.bincount();
    unordered_map<int, unordered_map<int, double>> jointCounts;
    double totalWeight = 0;
    for (auto i = 0; i < numSamples; i++) {
        jointCounts[secondFeature[i].item<int>()][firstFeature[i].item<int>()] += 1;
        totalWeight += 1;
    }
    if (totalWeight == 0)
        throw invalid_argument("Total weight should not be zero");
    double entropy = 0;
    for (int value = 0; value < featureCounts.sizes()[0]; ++value) {
        double p_f = featureCounts[value].item<double>() / totalWeight;
        double entropy_f = 0;
        for (auto& [label, jointCount] : jointCounts[value]) {
            double p_l_f = jointCount / featureCounts[value].item<double>();
            if (p_l_f > 0) {
                entropy_f -= p_l_f * log2(p_l_f);
            } else {
                entropy_f = 0;
            }
        }
        entropy += p_f * entropy_f;
    }
    return entropy;
}

// I(X;Y) = H(Y) - H(Y|X)
double mutualInformation(torch::Tensor firstFeature, torch::Tensor secondFeature)
{
    return entropy(firstFeature) - conditionalEntropy(firstFeature, secondFeature);
}
double entropy2(torch::Tensor feature)
{
    return torch::special::entr(feature).sum().item<double>();
}
int main()
{
    //int i = 3, j = 1, k = 2; // Indices for the cell you want to update
    // Print original tensor
    // torch::Tensor t = torch::tensor({ {1, 2, 3}, {4, 5, 6} }); // 3D tensor for this example
    // auto variables = vector<string>{ "A", "B" };
    // auto cardinalities = vector<int>{ 5, 4 };
    // torch::Tensor values = torch::rand({ 5, 4 });
    // auto candidate = "B";
    // vector<string> newVariables;
    // vector<int> newCardinalities;
    // for (int i = 0; i < variables.size(); i++) {
    //     if (variables[i] != candidate) {
    //         newVariables.push_back(variables[i]);
    //         newCardinalities.push_back(cardinalities[i]);
    //     }
    // }
    // torch::Tensor newValues = values.sum(1);
    // cout << "original values" << endl;
    // cout << values << endl;
    // cout << "newValues" << endl;
    // cout << newValues << endl;
    // cout << "newVariables" << endl;
    // for (auto& variable : newVariables) {
    //     cout << variable << endl;
    // }
    // cout << "newCardinalities" << endl;
    // for (auto& cardinality : newCardinalities) {
    //     cout << cardinality << endl;
    // }
    // auto row2 = values.index({ torch::tensor(1) }); // 
    // cout << "row2" << endl;
    // cout << row2 << endl;
    // auto col2 = values.index({ "...", 1 });
    // cout << "col2" << endl;
    // cout << col2 << endl;
    // auto col_last = values.index({ "...", -1 });
    // cout << "col_last" << endl;
    // cout << col_last << endl;
    // values.index_put_({ "...", -1 }, torch::tensor({ 1,2,3,4,5 }));
    // cout << "col_last" << endl;
    // cout << col_last << endl;
    // auto slice2 = values.index({ torch::indexing::Slice(1, torch::indexing::None) });
    // cout << "slice2" << endl;
    // cout << slice2 << endl;
    // auto mask = values.index({ "...", -1 }) % 2 == 0;
    // auto filter = values.index({ mask, 2 }); // Filter values
    // cout << "filter" << endl;
    // cout << filter << endl;
    // torch::Tensor dataset = torch::tensor({ {1,0,0,1},{1,1,1,2},{0,0,0,1},{1,0,2,0},{0,0,3,0} });
    // cout << "dataset" << endl;
    // cout << dataset << endl;
    // cout << "entropy(dataset.indices('...', 2))" << endl;
    // cout << dataset.index({ "...", 2 }) << endl;
    // cout << "*********************************" << endl;
    // for (int i = 0; i < 4; i++) {
    //     cout << "datset(" << i << ")" << endl;
    //     cout << dataset.index({ "...", i }) << endl;
    //     cout << "entropy(" << i << ")" << endl;
    //     cout << entropy(dataset.index({ "...", i })) << endl;
    // }
    // cout << "......................................" << endl;
    // //cout << entropy2(dataset.index({ "...", 2 }));
    // cout << "conditional entropy 0 2" << endl;
    // cout << conditionalEntropy(dataset.index({ "...", 0 }), dataset.index({ "...", 2 })) << endl;
    // cout << "mutualInformation(dataset.index({ '...', 0 }), dataset.index({ '...', 2 }))" << endl;
    // cout << mutualInformation(dataset.index({ "...", 0 }), dataset.index({ "...", 2 })) << endl;
    // auto test = torch::tensor({ .1, .2, .3 }, torch::kFloat);
    // auto result = torch::zeros({ 3, 3 }, torch::kFloat);
    // result.index_put_({ indices }, test);
    // cout << "indices" << endl;
    // cout << indices << endl;
    // cout << "result" << endl;
    // cout << result << endl;
    // cout << "Test" << endl;
    // cout << torch::triu(test.reshape(3, 3), torch::kFloat)) << endl;


    // Create a 3x3 tensor with zeros
    torch::Tensor tensor_3x3 = torch::zeros({ 3, 3 }, torch::kFloat);

    // Create a 1D tensor with the three elements you want to set in the upper corner
    torch::Tensor tensor_1d = torch::tensor({ 10, 11, 12 }, torch::kFloat);

    // Set the upper corner of the 3x3 tensor
    auto indices = torch::triu_indices(3, 3, 1);
    for (auto i = 0; i < tensor_1d.sizes()[0]; ++i) {
        auto x = indices[0][i];
        auto y = indices[1][i];
        tensor_3x3[x][y] = tensor_1d[i];
        tensor_3x3[y][x] = tensor_1d[i];
    }
    // Print the resulting 3x3 tensor
    std::cout << tensor_3x3 << std::endl;
    vector<int> v = { 1,2,3,4,5 };
    torch::Tensor t = torch::tensor(v);
    cout << t << endl;






    // std::cout << t << std::endl;
    // std::cout << "sum(0)" << std::endl;
    // std::cout << t.sum(0) << std::endl;
    // std::cout << "sum(1)" << std::endl;
    // std::cout << t.sum(1) << std::endl;
    // std::cout << "Normalized" << std::endl;
    // std::cout << t / t.sum(0) << std::endl;

    // New value
    // torch::Tensor new_val = torch::tensor(10.0f);

    // // Indices for the cell you want to update
    // std::vector<torch::Tensor> indices;
    // indices.push_back(torch::tensor(i)); // Replace i with your index for the 1st dimension
    // indices.push_back(torch::tensor(j)); // Replace j with your index for the 2nd dimension
    // indices.push_back(torch::tensor(k)); // Replace k with your index for the 3rd dimension
    // //torch::ArrayRef<at::indexing::TensorIndex> indices_ref(indices);
    // // Update cell
    // //torch::Tensor result = torch::stack(indices);
    // //torch::List<c10::optional<torch::Tensor>> indices_list = { torch::tensor(i), torch::tensor(j), torch::tensor(k) };
    // torch::List<c10::optional<torch::Tensor>> indices_list;
    // indices_list.push_back(torch::tensor(i));
    // indices_list.push_back(torch::tensor(j));
    // indices_list.push_back(torch::tensor(k));
    // //t.index_put_({ torch::tensor(i), torch::tensor(j), torch::tensor(k) }, new_val);
    // t.index_put_(indices_list, new_val);

    // // Print updated tensor
    // std::cout << t << std::endl;
}
