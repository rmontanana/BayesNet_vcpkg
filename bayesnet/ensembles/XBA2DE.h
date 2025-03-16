// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2025 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef XBA2DE_H
#define XBA2DE_H
#include <string>
#include <vector>
#include "Boost.h"
namespace bayesnet {
    class XBA2DE : public Boost {
    public:
        explicit XBA2DE(bool predict_voting = false);
        virtual ~XBA2DE() = default;
        std::vector<std::string> graph(const std::string& title = "XBA2DE") const override;
        std::string getVersion() override { return version; };
    protected:
        void trainModel(const torch::Tensor& weights, const Smoothing_t smoothing) override;
    private:
        std::vector<int> initializeModels(const Smoothing_t smoothing);
        std::vector<std::vector<int>> X_train_, X_test_;
        std::vector<int> y_train_, y_test_;
        std::string version = "0.9.7";
    };
}
#endif
