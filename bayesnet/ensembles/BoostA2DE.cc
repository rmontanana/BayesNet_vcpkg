// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <set>
#include <functional>
#include <limits.h>
#include <tuple>
#include <folding.hpp>
#include "bayesnet/feature_selection/CFS.h"
#include "bayesnet/feature_selection/FCBF.h"
#include "bayesnet/feature_selection/IWSS.h"
#include "BoostA2DE.h"

namespace bayesnet {

    BoostA2DE::BoostA2DE(bool predict_voting) : Boost(predict_voting)
    {
    }
    void BoostA2DE::trainModel(const torch::Tensor& weights)
    {

    }
    std::vector<std::string> BoostA2DE::graph(const std::string& title) const
    {
        return Ensemble::graph(title);
    }
}