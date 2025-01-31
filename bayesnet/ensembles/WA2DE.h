// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************
#ifndef WA2DE_H
#define WA2DE_H
#include "Ensemble.h"
#include <torch/torch.h>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>
namespace bayesnet {
    /**
     * Geoffrey I. Webb's A2DE (Averaged 2-Dependence Estimators) classifier
     * Implements the A2DE algorithm as an ensemble of SPODE models.
     */
    class WA2DE : public Ensemble {
    public:
        explicit WA2DE(bool predict_voting = false);
        virtual ~WA2DE() {};

        // Override method to set hyperparameters
        void setHyperparameters(const nlohmann::json& hyperparameters) override;

        // Graph visualization function
        std::vector<std::string> graph(const std::string& title = "A2DE") const override;
        torch::Tensor computeProbabilities(const torch::Tensor& data) const;
        double score(const torch::Tensor& X, const torch::Tensor& y);
    protected:
        // Model-building function
        void buildModel(const torch::Tensor& weights) override;

    private:
        int num_classes_;                // Number of classes
        int num_attributes_;             // Number of attributes
        std::vector<int> attribute_cardinalities_; // Cardinalities of attributes

        // Frequency counts (similar to Java implementation)
        std::vector<double> class_counts_;  // Class frequency
        std::vector<std::vector<std::vector<double>>> freq_attr_class_; // P(A | C)
        std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>> freq_pair_class_; // P(A_i, A_j | C)

        double total_count_; // Total instance count

        bool weighted_a2de_; // Whether to use weighted A2DE
        double smoothing_factor_; // Smoothing parameter (default: Laplace)
        torch::Tensor AODEConditionalProb(const torch::Tensor& data);
        void trainModel(const torch::Tensor& data, const Smoothing_t smoothing);
        int toIntValue(int attributeIndex, float value) const;
    };
}
#endif