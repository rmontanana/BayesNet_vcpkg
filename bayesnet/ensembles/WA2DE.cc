// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************
#include "WA2DE.h"
namespace bayesnet {
    WA2DE::WA2DE(bool predict_voting)
        : num_classes_(0), num_attributes_(0), total_count_(0.0), weighted_a2de_(false), smoothing_factor_(1.0)
    {
        validHyperparameters = { "predict_voting" };
        std::cout << "WA2DE classifier created.\n";
    }

    void bayesnet::WA2DE::setHyperparameters(const nlohmann::json& hyperparameters_)
    {
        auto hyperparameters = hyperparameters_;
        if (hyperparameters.contains("predict_voting")) {
            predict_voting = hyperparameters["predict_voting"];
            hyperparameters.erase("predict_voting");
        }
        Classifier::setHyperparameters(hyperparameters);
    }


    void WA2DE::buildModel(const torch::Tensor& weights)
    {
        for (int c = 0; c < num_classes_; ++c) {
            class_counts_[c] += 1e-4; // Laplace smoothing
        }
        for (int a = 0; a < num_attributes_; ++a) {
            for (int v = 0; v < attribute_cardinalities_[a]; ++v) {
                for (int c = 0; c < num_classes_; ++c) {
                    freq_attr_class_[a][v][c] =
                        (freq_attr_class_[a][v][c] + 1.0) / (class_counts_[c] + attribute_cardinalities_[a]);
                }
            }
        }

        for (int sp = 0; sp < num_attributes_; ++sp) {
            for (int spv = 0; spv < attribute_cardinalities_[sp]; ++spv) {
                for (int ch = 0; ch < num_attributes_; ++ch) {
                    if (sp != ch) {
                        for (int chv = 0; chv < attribute_cardinalities_[ch]; ++chv) {
                            for (int c = 0; c < num_classes_; ++c) {
                                freq_pair_class_[sp][spv][ch][chv][c] =
                                    (freq_pair_class_[sp][spv][ch][chv][c] + 1.0) /
                                    (class_counts_[c] + attribute_cardinalities_[sp] * attribute_cardinalities_[ch]);
                            }
                        }
                    }
                }
            }
        }
        std::cout << "Model probabilities computed.\n";
    }
    void WA2DE::trainModel(const torch::Tensor& weights, const Smoothing_t smoothing)
    {
        auto data = dataset.clone();
        auto labels = data[-1];
        // Remove class row from data
        data = data.index({ at::indexing::Slice(0, -1) });
        std::cout << "Training A2DE model...\n";
        std::cout << "Data: " << data.sizes() << std::endl;
        std::cout << "Labels: " << labels.sizes() << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        if (data.dim() != 2 || labels.dim() != 1) {
            throw std::invalid_argument("Invalid input dimensions.");
        }
        num_attributes_ = data.size(0);
        num_classes_ = labels.max().item<int>() + 1;
        total_count_ = data.size(1);
        std::cout << "Number of attributes: " << num_attributes_ << std::endl;
        std::cout << "Number of classes: " << num_classes_ << std::endl;
        std::cout << "Total count: " << total_count_ << std::endl;

        // Compute cardinalities
        attribute_cardinalities_.clear();
        for (int i = 0; i < num_attributes_; ++i) {
            attribute_cardinalities_.push_back(data[i].max().item<int>() + 1);
        }
        std::cout << "Attribute cardinalities: ";
        for (int i = 0; i < num_attributes_; ++i) {
            std::cout << attribute_cardinalities_[i] << " ";
        }
        std::cout << std::endl;
        // output the map of states
        std::cout << "States: ";
        for (int i = 0; i < states.size() - 1; i++) {
            std::cout << features[i] << " " << states[features[i]].size() << std::endl;
        }

        // Resize storage
        class_counts_.resize(num_classes_, 0.0);
        freq_attr_class_.resize(num_attributes_);
        freq_pair_class_.resize(num_attributes_);

        for (int i = 0; i < num_attributes_; ++i) {
            freq_attr_class_[i].resize(attribute_cardinalities_[i], std::vector<double>(num_classes_, 0.0));
            freq_pair_class_[i].resize(attribute_cardinalities_[i]); // Ensure first level exists
            for (int j = 0; j < attribute_cardinalities_[i]; ++j) {
                freq_pair_class_[i][j].resize(num_attributes_); // Ensure second level exists
                for (int k = 0; k < num_attributes_; ++k) {
                    if (i != k) {
                        freq_pair_class_[i][j][k].resize(attribute_cardinalities_[k]); // Ensure third level exists
                        for (int l = 0; l < attribute_cardinalities_[k]; ++l) {
                            freq_pair_class_[i][j][k][l].resize(num_classes_, 0.0); // Finally, initialize with 0.0
                        }
                    }
                }
            }
        }
        // Count frequencies
        auto data_cpu = data.to(torch::kCPU);
        auto labels_cpu = labels.to(torch::kCPU);
        int32_t* data_ptr = data_cpu.data_ptr<int32_t>();
        int32_t* labels_ptr = labels_cpu.data_ptr<int32_t>();

        for (int i = 0; i < total_count_; ++i) {
            int class_label = labels_ptr[i];
            class_counts_[class_label] += 1.0;

            std::vector<int> attr_values(num_attributes_);
            for (int a = 0; a < num_attributes_; ++a) {
                attr_values[a] = toIntValue(a, data_ptr[i * num_attributes_ + a]);
                freq_attr_class_[a][attr_values[a]][class_label] += 1.0;
            }

            // Pairwise counts
            for (int sp = 0; sp < num_attributes_; ++sp) {
                for (int ch = 0; ch < num_attributes_; ++ch) {
                    if (sp != ch) {
                        freq_pair_class_[sp][attr_values[sp]][ch][attr_values[ch]][class_label] += 1.0;
                    }
                }
            }
        }
        std::cout << "Verifying Frequency Counts:\n";
        for (int c = 0; c < num_classes_; ++c) {
            std::cout << "Class " << c << " Count: " << class_counts_[c] << std::endl;
        }

        for (int a = 0; a < num_attributes_; ++a) {
            for (int v = 0; v < attribute_cardinalities_[a]; ++v) {
                std::cout << "P(A[" << a << "]=" << v << "|C): ";
                for (int c = 0; c < num_classes_; ++c) {
                    std::cout << freq_attr_class_[a][v][c] << " ";
                }
                std::cout << std::endl;
            }
        }

    }

    torch::Tensor WA2DE::computeProbabilities(const torch::Tensor& data) const
    {
        int M = data.size(1);
        auto output = torch::zeros({ M, num_classes_ }, torch::kF64);

        auto data_cpu = data.to(torch::kCPU);
        int32_t* data_ptr = data_cpu.data_ptr<int32_t>();

        for (int i = 0; i < M; ++i) {
            std::vector<int> attr_values(num_attributes_);
            for (int a = 0; a < num_attributes_; ++a) {
                attr_values[a] = toIntValue(a, data_ptr[i * num_attributes_ + a]);
            }

            std::vector<double> log_prob(num_classes_, 0.0);
            for (int c = 0; c < num_classes_; ++c) {
                log_prob[c] = std::log((class_counts_[c] + smoothing_factor_) / (total_count_ + num_classes_ * smoothing_factor_));

                double sum_log = 0.0;
                for (int sp = 0; sp < num_attributes_; ++sp) {
                    double sp_log = log_prob[c];
                    for (int ch = 0; ch < num_attributes_; ++ch) {
                        if (sp == ch) continue;
                        double num = freq_pair_class_[sp][attr_values[sp]][ch][attr_values[ch]][c] + smoothing_factor_;
                        double denom = class_counts_[c] + attribute_cardinalities_[sp] * attribute_cardinalities_[ch] * smoothing_factor_;
                        sp_log += std::log(num / denom);
                    }
                    sum_log += std::exp(sp_log);
                }
                log_prob[c] = std::log(sum_log / num_attributes_);
            }

            double max_log = *std::max_element(log_prob.begin(), log_prob.end());
            double sum_exp = 0.0;
            for (int c = 0; c < num_classes_; ++c) {
                sum_exp += std::exp(log_prob[c] - max_log);
            }
            double log_sum_exp = max_log + std::log(sum_exp);

            for (int c = 0; c < num_classes_; ++c) {
                output[i][c] = std::exp(log_prob[c] - log_sum_exp);
            }
        }

        return output.to(torch::kF32);
    }
    int WA2DE::toIntValue(int attributeIndex, float value) const
    {
        int v = static_cast<int>(value);
        return std::max(0, std::min(v, attribute_cardinalities_[attributeIndex] - 1));
    }
    torch::Tensor WA2DE::AODEConditionalProb(const torch::Tensor& data)
    {
        int M = data.size(1);  // Number of test samples
        torch::Tensor output = torch::zeros({ M, num_classes_ }, torch::kF32);

        auto data_cpu = data.to(torch::kCPU);
        int32_t* data_ptr = data_cpu.data_ptr<int32_t>();

        for (int i = 0; i < M; ++i) {
            std::vector<int> attr_values(num_attributes_);
            for (int a = 0; a < num_attributes_; ++a) {
                attr_values[a] = toIntValue(a, data_ptr[i * num_attributes_ + a]);
            }

            std::vector<double> log_prob(num_classes_, 0.0);
            for (int c = 0; c < num_classes_; ++c) {
                log_prob[c] = std::log(class_counts_[c] / total_count_);

                double sum_log = 0.0;
                for (int sp = 0; sp < num_attributes_; ++sp) {
                    double sp_log = log_prob[c];
                    for (int ch = 0; ch < num_attributes_; ++ch) {
                        if (sp == ch) continue;
                        double prob = freq_pair_class_[sp][attr_values[sp]][ch][attr_values[ch]][c];
                        sp_log += std::log(prob);
                    }
                    sum_log += std::exp(sp_log);
                }
                log_prob[c] = std::log(sum_log / num_attributes_);
            }

            double max_log = *std::max_element(log_prob.begin(), log_prob.end());
            double sum_exp = 0.0;
            for (int c = 0; c < num_classes_; ++c) {
                sum_exp += std::exp(log_prob[c] - max_log);
            }
            double log_sum_exp = max_log + std::log(sum_exp);

            for (int c = 0; c < num_classes_; ++c) {
                output[i][c] = std::exp(log_prob[c] - log_sum_exp);
            }
        }

        return output;
    }

    double WA2DE::score(const torch::Tensor& X, const torch::Tensor& y)
    {
        torch::Tensor preds = AODEConditionalProb(X);
        torch::Tensor pred_labels = preds.argmax(1);

        auto correct = pred_labels.eq(y).sum().item<int>();
        auto total = y.size(0);

        return static_cast<double>(correct) / total;
    }

    std::vector<std::string> WA2DE::graph(const std::string& title) const
    {
        return { title, "Graph visualization not implemented." };
    }
}