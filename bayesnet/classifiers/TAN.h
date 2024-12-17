// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef TAN_H
#define TAN_H
#include "Classifier.h"
namespace bayesnet {
    class TAN : public Classifier {
    public:
        TAN();
        virtual ~TAN() = default;
        void setHyperparameters(const nlohmann::json& hyperparameters_) override;
        std::vector<std::string> graph(const std::string& name = "TAN") const override;
    protected:
        void buildModel(const torch::Tensor& weights) override;
    private:
        int parent = -1;
    };
}
#endif