// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef BOOSTA2DE_H
#define BOOSTA2DE_H
#include <string>
#include <vector>
#include "bayesnet/classifiers/SPnDE.h"
#include "Boost.h"
namespace bayesnet {
    class BoostA2DE : public Boost {
    public:
        explicit BoostA2DE(bool predict_voting = false);
        virtual ~BoostA2DE() = default;
        std::vector<std::string> graph(const std::string& title = "BoostA2DE") const override;
    protected:
        void trainModel(const torch::Tensor& weights) override;
    };
}
#endif