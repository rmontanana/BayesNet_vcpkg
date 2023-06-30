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

int main()
{
    torch::Tensor t = torch::rand({ 5, 4, 3 }); // 3D tensor for this example
    int i = 3, j = 1, k = 2; // Indices for the cell you want to update
    // Print original tensor
    std::cout << t << std::endl;

    // New value
    torch::Tensor new_val = torch::tensor(10.0f);

    // Indices for the cell you want to update
    std::vector<torch::Tensor> indices;
    indices.push_back(torch::tensor(i)); // Replace i with your index for the 1st dimension
    indices.push_back(torch::tensor(j)); // Replace j with your index for the 2nd dimension
    indices.push_back(torch::tensor(k)); // Replace k with your index for the 3rd dimension
    //torch::ArrayRef<at::indexing::TensorIndex> indices_ref(indices);
    // Update cell
    //torch::Tensor result = torch::stack(indices);
    //torch::List<c10::optional<torch::Tensor>> indices_list = { torch::tensor(i), torch::tensor(j), torch::tensor(k) };
    torch::List<c10::optional<torch::Tensor>> indices_list;
    indices_list.push_back(torch::tensor(i));
    indices_list.push_back(torch::tensor(j));
    indices_list.push_back(torch::tensor(k));
    //t.index_put_({ torch::tensor(i), torch::tensor(j), torch::tensor(k) }, new_val);
    t.index_put_(indices_list, new_val);

    // Print updated tensor
    std::cout << t << std::endl;
}
