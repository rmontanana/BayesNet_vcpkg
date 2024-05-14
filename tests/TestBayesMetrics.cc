// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "bayesnet/utils/BayesMetrics.h"
#include "TestUtils.h"


TEST_CASE("Metrics Test", "[Metrics]")
{
    std::string file_name = GENERATE("glass", "iris", "ecoli", "diabetes");
    map<std::string, pair<int, std::vector<int>>> resultsKBest = {
        {"glass", {7, { 0, 1, 7, 6, 3, 5, 2 }}},
        {"iris", {3, { 0, 3, 2 }} },
        {"ecoli", {6, { 2, 4, 1, 0, 6, 5 }}},
        {"diabetes", {2, { 7, 1 }}}
    };
    map<std::string, double> resultsMI = {
        {"glass", 0.12805398},
        {"iris", 0.3158139948},
        {"ecoli", 0.0089431099},
        {"diabetes", 0.0345470614}
    };
    map<pair<std::string, int>, std::vector<pair<int, int>>> resultsMST = {
        { {"glass", 0}, { {0, 6}, {0, 5}, {0, 3}, {5, 1}, {5, 8}, {5, 4}, {6, 2}, {6, 7} } },
        { {"glass", 1}, { {1, 5}, {5, 0}, {5, 8}, {5, 4}, {0, 6}, {0, 3}, {6, 2}, {6, 7} } },
        { {"iris", 0}, { {0, 1}, {0, 2}, {1, 3} } },
        { {"iris", 1}, { {1, 0}, {1, 3}, {0, 2} } },
        { {"ecoli", 0}, { {0, 1}, {0, 2}, {1, 5}, {1, 3}, {5, 6}, {5, 4} } },
        { {"ecoli", 1}, { {1, 0}, {1, 5}, {1, 3}, {5, 6}, {5, 4}, {0, 2} } },
        { {"diabetes", 0}, { {0, 7}, {0, 2}, {0, 6}, {2, 3}, {3, 4}, {3, 5}, {4, 1} } },
        { {"diabetes", 1}, { {1, 4}, {4, 3}, {3, 2}, {3, 5}, {2, 0}, {0, 7}, {0, 6} } }
    };
    auto raw = RawDatasets(file_name, true);
    bayesnet::Metrics metrics(raw.dataset, raw.features, raw.className, raw.classNumStates);
    bayesnet::Metrics metricsv(raw.Xv, raw.yv, raw.features, raw.className, raw.classNumStates);

    SECTION("Test Constructor")
    {
        REQUIRE(metrics.getScoresKBest().size() == 0);
        REQUIRE(metricsv.getScoresKBest().size() == 0);
    }

    SECTION("Test SelectKBestWeighted")
    {
        std::vector<int> kBest = metrics.SelectKBestWeighted(raw.weights, true, resultsKBest.at(file_name).first);
        std::vector<int> kBestv = metricsv.SelectKBestWeighted(raw.weights, true, resultsKBest.at(file_name).first);
        REQUIRE(kBest.size() == resultsKBest.at(file_name).first);
        REQUIRE(kBestv.size() == resultsKBest.at(file_name).first);
        REQUIRE(kBest == resultsKBest.at(file_name).second);
        REQUIRE(kBestv == resultsKBest.at(file_name).second);
    }

    SECTION("Test Mutual Information")
    {
        auto result = metrics.mutualInformation(raw.dataset.index({ 1, "..." }), raw.dataset.index({ 2, "..." }), raw.weights);
        auto resultv = metricsv.mutualInformation(raw.dataset.index({ 1, "..." }), raw.dataset.index({ 2, "..." }), raw.weights);
        REQUIRE(result == Catch::Approx(resultsMI.at(file_name)).epsilon(raw.epsilon));
        REQUIRE(resultv == Catch::Approx(resultsMI.at(file_name)).epsilon(raw.epsilon));
    }

    SECTION("Test Maximum Spanning Tree")
    {
        auto weights_matrix = metrics.conditionalEdge(raw.weights);
        auto weights_matrixv = metricsv.conditionalEdge(raw.weights);
        for (int i = 0; i < 2; ++i) {
            auto result = metrics.maximumSpanningTree(raw.features, weights_matrix, i);
            auto resultv = metricsv.maximumSpanningTree(raw.features, weights_matrixv, i);
            REQUIRE(result == resultsMST.at({ file_name, i }));
            REQUIRE(resultv == resultsMST.at({ file_name, i }));
        }
    }
}
TEST_CASE("Select all features ordered by Mutual Information", "[Metrics]")
{
    auto raw = RawDatasets("iris", true);
    bayesnet::Metrics metrics(raw.dataset, raw.features, raw.className, raw.classNumStates);
    auto kBest = metrics.SelectKBestWeighted(raw.weights, true, 0);
    REQUIRE(kBest.size() == raw.features.size());
    REQUIRE(kBest == std::vector<int>({ 1, 0, 3, 2 }));
}
TEST_CASE("Entropy Test", "[Metrics]")
{
    auto raw = RawDatasets("iris", true);
    bayesnet::Metrics metrics(raw.dataset, raw.features, raw.className, raw.classNumStates);
    auto result = metrics.entropy(raw.dataset.index({ 0, "..." }), raw.weights);
    REQUIRE(result == Catch::Approx(0.9848175048828125).epsilon(raw.epsilon));
    auto data = torch::tensor({ 0, 0, 0, 0, 0, 0, 0, 1, 1, 1 }, torch::kInt32);
    auto weights = torch::tensor({ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, torch::kFloat32);
    result = metrics.entropy(data, weights);
    REQUIRE(result == Catch::Approx(0.61086434125900269).epsilon(raw.epsilon));
    data = torch::tensor({ 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 }, torch::kInt32);
    result = metrics.entropy(data, weights);
    REQUIRE(result == Catch::Approx(0.693147180559945).epsilon(raw.epsilon));
}
TEST_CASE("Conditional Entropy", "[Metrics]")
{
    auto raw = RawDatasets("iris", true);
    bayesnet::Metrics metrics(raw.dataset, raw.features, raw.className, raw.classNumStates);
    auto feature0 = raw.dataset.index({ 0, "..." });
    auto feature1 = raw.dataset.index({ 1, "..." });
    auto feature2 = raw.dataset.index({ 2, "..." });
    auto feature3 = raw.dataset.index({ 3, "..." });
    auto labels = raw.dataset.index({ 4, "..." });
    auto result = metrics.conditionalEntropy(feature0, feature1, labels, raw.weights);
    auto result2 = metrics.conditionalEntropy2(feature0, feature1, labels, raw.weights);
    std::cout << "Result=" << result << "\n";
    std::cout << "Result2=" << result2 << "\n";
}