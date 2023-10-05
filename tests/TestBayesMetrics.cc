#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "BayesMetrics.h"
#include "TestUtils.h"

using namespace std;

TEST_CASE("Metrics Test", "[Metrics]")
{
    string file_name = GENERATE("glass", "iris", "ecoli", "diabetes");
    map<string, pair<int, vector<int>>> resultsKBest = {
        {"glass", {7, { 3, 2, 0, 1, 6, 7, 5 }}},
        {"iris", {3, { 1, 0, 2 }} },
        {"ecoli", {6, { 2, 3, 1, 0, 4, 5 }}},
        {"diabetes", {2, { 2, 0 }}}
    };
    auto [XDisc, yDisc, featuresDisc, classNameDisc, statesDisc] = loadDataset(file_name, true, true);
    int classNumStates = statesDisc.at(classNameDisc).size();
    auto yresized = torch::transpose(yDisc.view({ yDisc.size(0), 1 }), 0, 1);
    torch::Tensor dataset = torch::cat({ XDisc, yresized }, 0);
    int nSamples = dataset.size(1);

    SECTION("Test Constructor")
    {
        bayesnet::Metrics metrics(XDisc, featuresDisc, classNameDisc, classNumStates);
        REQUIRE(metrics.getScoresKBest().size() == 0);
    }

    SECTION("Test SelectKBestWeighted")
    {
        bayesnet::Metrics metrics(XDisc, featuresDisc, classNameDisc, classNumStates);
        torch::Tensor weights = torch::full({ nSamples }, 1.0 / nSamples, torch::kDouble);
        vector<int> kBest = metrics.SelectKBestWeighted(weights, true, resultsKBest.at(file_name).first);
        REQUIRE(kBest.size() == resultsKBest.at(file_name).first);
        REQUIRE(kBest == resultsKBest.at(file_name).second);
    }

    SECTION("Test mutualInformation")
    {
        bayesnet::Metrics metrics(XDisc, featuresDisc, classNameDisc, classNumStates);
        torch::Tensor weights = torch::full({ nSamples }, 1.0 / nSamples, torch::kDouble);
        auto result = metrics.mutualInformation(dataset.index({ 1, "..." }), dataset.index({ 2, "..." }), weights);
        REQUIRE(result == 0.87);
    }
}