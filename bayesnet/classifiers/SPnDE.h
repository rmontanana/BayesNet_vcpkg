// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef SPnDE_H
#define SPnDE_H
#include <vector>
#include "Classifier.h"

namespace bayesnet {
    class SPnDE : public Classifier {
    public:
        explicit SPnDE(std::vector<int> parents);
        virtual ~SPnDE() = default;
        std::vector<std::string> graph(const std::string& name = "SPnDE") const override;
    protected:
        void buildModel(const torch::Tensor& weights) override;
    private:
        std::vector<int> parents;


    };
}
#endif