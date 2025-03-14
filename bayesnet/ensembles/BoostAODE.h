// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef BOOSTAODE_H
#define BOOSTAODE_H
#include <string>
#include <vector>
#include "Boost.h"

namespace bayesnet {
    class BoostAODE : public Boost {
    public:
        explicit BoostAODE(bool predict_voting = false);
        virtual ~BoostAODE() = default;
        std::vector<std::string> graph(const std::string& title = "BoostAODE") const override;
    protected:
        void trainModel(const torch::Tensor& weights, const Smoothing_t smoothing) override;
    private:
        std::vector<int> initializeModels(const Smoothing_t smoothing);
    };
}
#endif
