// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef A2DE_H
#define A2DE_H
#include "bayesnet/classifiers/SPnDE.h"
#include "Ensemble.h"
namespace bayesnet {
    class A2DE : public Ensemble {
    public:
        A2DE(bool predict_voting = false);
        virtual ~A2DE() {};
        void setHyperparameters(const nlohmann::json& hyperparameters) override;
        std::vector<std::string> graph(const std::string& title = "A2DE") const override;
    protected:
        void buildModel(const torch::Tensor& weights) override;
    };
}
#endif