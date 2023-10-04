#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "BayesMetrics.h"
#include "TestUtils.h"

using namespace std;

TEST_CASE("Metrics Test", "[Metrics]")
{
    string file_name = GENERATE("glass", "iris", "ecoli", "diabetes");
    map<string, pair<int, vector<int>>> results = {
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
        vector<int> kBest = metrics.SelectKBestWeighted(weights, true, results.at(file_name).first);
        REQUIRE(kBest.size() == results.at(file_name).first);
        REQUIRE(kBest == results.at(file_name).second);
    }

    SECTION("Test mutualInformation")
    {
        // torch::Tensor samples = torch::rand({ 10, 5 });
        // vector<string> features = { "feature1", "feature2", "feature3", "feature4", "feature5" };
        // string className = "class1";
        // int classNumStates = 2;

        // bayesnet::Metrics obj(samples, features, className, classNumStates);

        // torch::Tensor firstFeature = samples.select(1, 0);
        // torch::Tensor secondFeature = samples.select(1, 1);
        // torch::Tensor weights = torch::ones({ 10 });

        // double mi = obj.mutualInformation(firstFeature, secondFeature, weights);

        // REQUIRE(mi >= 0);
    }
}