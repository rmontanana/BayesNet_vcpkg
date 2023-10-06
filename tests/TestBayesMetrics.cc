#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "BayesMetrics.h"
#include "TestUtils.h"

using namespace std;

TEST_CASE("Metrics Test", "[BayesNet]")
{
    // string file_name = GENERATE("glass", "iris", "ecoli", "diabetes");
    string file_name = "glass";
    //
    //
    // OJO CAMBIAR
    //
    //
    map<string, pair<int, vector<int>>> resultsKBest = {
        {"glass", {7, { 0, 1, 7, 6, 3, 5, 2 }}},
        {"iris", {3, { 0, 3, 2 }} },
        {"ecoli", {6, { 2, 4, 1, 0, 6, 5 }}},
        {"diabetes", {2, { 7, 1 }}}
    };
    map<string, double> resultsMI = {
        {"glass", 0.12805398},
        {"iris", 0.3158139948},
        {"ecoli", 0.0089431099},
        {"diabetes", 0.0345470614}
    };
    map<string, vector<pair<int, int>>> resultsMST = {
        //{"glass", {{0,6}, {0,5}, {0,3}, {6,2}, {6,7}, {5,1}, {5,8}, {5,4}}},
        {"glass", {{0,6}, {0,5}, {0,3}, {5,1}, {5,8}, {5,4}, {6,2}, {6,7}}},
        {"iris", {{0,1},{0,2},{1,3}}},
        {"ecoli", {{0,1}, {0,2}, {1,5}, {1,3}, {5,6}, {5,4}}},
        {"diabetes", {{0,7}, {0,2}, {0,6}, {2,3}, {3,4}, {3,5}, {4,1}}}
    };
    auto raw = RawDatasets(file_name, true);
    bayesnet::Metrics metrics(raw.dataset, raw.featurest, raw.classNamet, raw.classNumStates);

    SECTION("Test Constructor")
    {
        REQUIRE(metrics.getScoresKBest().size() == 0);
    }

    SECTION("Test SelectKBestWeighted")
    {
        vector<int> kBest = metrics.SelectKBestWeighted(raw.weights, true, resultsKBest.at(file_name).first);
        REQUIRE(kBest.size() == resultsKBest.at(file_name).first);
        REQUIRE(kBest == resultsKBest.at(file_name).second);
    }

    SECTION("Test Mutual Information")
    {
        auto result = metrics.mutualInformation(raw.dataset.index({ 1, "..." }), raw.dataset.index({ 2, "..." }), raw.weights);
        REQUIRE(result == Catch::Approx(resultsMI.at(file_name)).epsilon(raw.epsilon));
    }

    SECTION("Test Maximum Spanning Tree")
    {
        auto weights_matrix = metrics.conditionalEdge(raw.weights);
        cout << "Weights matrix: " << endl;
        for (int i = 0; i < static_cast<int>(raw.featurest.size()); ++i) {
            for (int j = 0; j < static_cast<int>(raw.featurest.size()); ++j) {
                cout << setw(10) << setprecision(7) << fixed << weights_matrix[i][j].item<float>() << ",  ";
            }
            cout << endl;
        }
        auto result = metrics.maximumSpanningTree(raw.featurest, weights_matrix, 0);
        REQUIRE(result == resultsMST.at(file_name));
    }
}