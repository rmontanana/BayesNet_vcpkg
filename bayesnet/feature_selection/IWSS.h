// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef IWSS_H
#define IWSS_H
#include <vector>
#include <torch/torch.h>
#include "FeatureSelect.h"
namespace bayesnet {
    class IWSS : public FeatureSelect {
    public:
        // dataset is a n+1xm tensor of integers where dataset[-1] is the y std::vector
        IWSS(const torch::Tensor& samples, const std::vector<std::string>& features, const std::string& className, const int maxFeatures, const int classNumStates, const torch::Tensor& weights, const double threshold);
        virtual ~IWSS() {};
        void fit() override;
    private:
        double threshold = -1;
    };
}
#endif