#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "bayesnet/utils/BayesMetrics.h"
#include "bayesnet/feature_selection/CFS.h"
#include "bayesnet/feature_selection/FCBF.h"
#include "bayesnet/feature_selection/IWSS.h"
#include "TestUtils.h"

bayesnet::FeatureSelect* build_selector(RawDatasets& raw, std::string selector, double threshold)
{
    if (selector == "CFS") {
        return new bayesnet::CFS(raw.dataset, raw.featuresv, raw.classNamev, raw.featuresv.size(), raw.classNumStates, raw.weights);
    } else if (selector == "FCBF") {
        return new bayesnet::FCBF(raw.dataset, raw.featuresv, raw.classNamev, raw.featuresv.size(), raw.classNumStates, raw.weights, threshold);
    } else if (selector == "IWSS") {
        return new bayesnet::IWSS(raw.dataset, raw.featuresv, raw.classNamev, raw.featuresv.size(), raw.classNumStates, raw.weights, threshold);
    }
    return nullptr;
}

TEST_CASE("Features Selected", "[FeatureSelection]")
{
    std::string file_name = GENERATE("glass", "iris", "ecoli", "diabetes");

    auto raw = RawDatasets(file_name, true);

    SECTION("Test features selected and size")
    {
        map<pair<std::string, std::string>, std::vector<int>> results = {
            { {"glass", "CFS"}, { 2, 3, 6, 1, 8, 4 } },
            { {"iris", "CFS"}, { 3, 2, 1, 0 } },
            { {"ecoli", "CFS"}, { 5, 0, 4, 2, 1, 6 } },
            { {"diabetes", "CFS"}, { 1, 5, 7, 6, 4, 2 } },
            { {"glass", "IWSS" }, { 2, 3, 5, 7, 6 } },
            { {"iris", "IWSS"}, { 3, 2, 0 } },
            { {"ecoli", "IWSS"}, { 5, 6, 0, 1, 4 } },
            { {"diabetes", "IWSS"}, { 1, 5, 4, 7, 3 } },
            { {"glass", "FCBF" }, { 2, 3, 5, 7, 6 } },
            { {"iris", "FCBF"}, { 3, 2 } },
            { {"ecoli", "FCBF"}, { 5, 0, 1, 4, 2 } },
            { {"diabetes", "FCBF"}, { 1, 5, 7, 6 } }
        };
        double threshold;
        std::string selector;
        std::vector<std::pair<std::string, double>> selectors = {
            { "CFS", 0.0 },
            { "IWSS", 0.5 },
            { "FCBF", 1e-7 }
        };
        for (const auto item : selectors) {
            selector = item.first; threshold = item.second;
            bayesnet::FeatureSelect* featureSelector = build_selector(raw, selector, threshold);
            featureSelector->fit();
            std::vector<int> selected = featureSelector->getFeatures();
            INFO("file_name: " << file_name << ", selector: " << selector);
            REQUIRE(selected.size() == results.at({ file_name, selector }).size());
            REQUIRE(selected == results.at({ file_name, selector }));
            delete featureSelector;
        }
    }
}

// TEST_CASE("Feature Selection Test", "[BayesNet]")
// {
//     std::string file_name = GENERATE("glass", "iris", "ecoli", "diabetes");
//     std::string selector = GENERATE("CFS", "FCBF", "IWSS");
//     map<std::string, pair<int, std::vector<int>>> resultsKBest = {
//         {"glass", {7, { 0, 1, 7, 6, 3, 5, 2 }}},
//         {"iris", {3, { 0, 3, 2 }} },
//         {"ecoli", {6, { 2, 4, 1, 0, 6, 5 }}},
//         {"diabetes", {2, { 7, 1 }}}
//     };
//     map<std::string, double> resultsMI = {
//         {"glass", 0.12805398},
//         {"iris", 0.3158139948},
//         {"ecoli", 0.0089431099},
//         {"diabetes", 0.0345470614}
//     };
//     map<pair<std::string, int>, std::vector<pair<int, int>>> resultsMST = {
//         { {"glass", 0}, { {0, 6}, {0, 5}, {0, 3}, {5, 1}, {5, 8}, {5, 4}, {6, 2}, {6, 7} } },
//         { {"glass", 1}, { {1, 5}, {5, 0}, {5, 8}, {5, 4}, {0, 6}, {0, 3}, {6, 2}, {6, 7} } },
//         { {"iris", 0}, { {0, 1}, {0, 2}, {1, 3} } },
//         { {"iris", 1}, { {1, 0}, {1, 3}, {0, 2} } },
//         { {"ecoli", 0}, { {0, 1}, {0, 2}, {1, 5}, {1, 3}, {5, 6}, {5, 4} } },
//         { {"ecoli", 1}, { {1, 0}, {1, 5}, {1, 3}, {5, 6}, {5, 4}, {0, 2} } },
//         { {"diabetes", 0}, { {0, 7}, {0, 2}, {0, 6}, {2, 3}, {3, 4}, {3, 5}, {4, 1} } },
//         { {"diabetes", 1}, { {1, 4}, {4, 3}, {3, 2}, {3, 5}, {2, 0}, {0, 7}, {0, 6} } }
//     };
//     auto raw = RawDatasets(file_name, true);
//     FeatureSelect* featureSelector = build_selector(raw, selector);

//     SECTION("Test Constructor")
//     {
//         REQUIRE(metrics.getScoresKBest().size() == 0);
//     }

//     SECTION("Test SelectKBestWeighted")
//     {
//         std::vector<int> kBest = metrics.SelectKBestWeighted(raw.weights, true, resultsKBest.at(file_name).first);
//         REQUIRE(kBest.size() == resultsKBest.at(file_name).first);
//         REQUIRE(kBest == resultsKBest.at(file_name).second);
//     }

//     SECTION("Test Mutual Information")
//     {
//         auto result = metrics.mutualInformation(raw.dataset.index({ 1, "..." }), raw.dataset.index({ 2, "..." }), raw.weights);
//         REQUIRE(result == Catch::Approx(resultsMI.at(file_name)).epsilon(raw.epsilon));
//     }

//     SECTION("Test Maximum Spanning Tree")
//     {
//         auto weights_matrix = metrics.conditionalEdge(raw.weights);
//         for (int i = 0; i < 2; ++i) {
//             auto result = metrics.maximumSpanningTree(raw.featurest, weights_matrix, i);
//             REQUIRE(result == resultsMST.at({ file_name, i }));
//         }
//     }
// }