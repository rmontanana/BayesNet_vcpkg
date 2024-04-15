// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef BAYESNET_UTILS_H
#define BAYESNET_UTILS_H
#include <vector>
#include <torch/torch.h>
namespace bayesnet {
    std::vector<int> argsort(std::vector<double>& nums);
    std::vector<std::vector<double>> tensorToVectorDouble(torch::Tensor& dtensor);
    torch::Tensor vectorToTensor(std::vector<std::vector<int>>& vector, bool transpose = true);
}
#endif //BAYESNET_UTILS_H