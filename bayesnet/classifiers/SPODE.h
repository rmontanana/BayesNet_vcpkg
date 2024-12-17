// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef SPODE_H
#define SPODE_H
#include "Classifier.h"

namespace bayesnet {
    class SPODE : public Classifier {
    public:
        explicit SPODE(int root);
        virtual ~SPODE() = default;
        void setHyperparameters(const nlohmann::json& hyperparameters_) override;
        std::vector<std::string> graph(const std::string& name = "SPODE") const override;
    protected:
        void buildModel(const torch::Tensor& weights) override;
    private:
        int root;
    };
}
#endif