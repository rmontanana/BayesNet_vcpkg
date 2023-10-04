#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "BayesMetrics.h"

using namespace std;

TEST_CASE("Metrics Test", "[Metrics]")
{
    SECTION("Test Constructor")
    {
        torch::Tensor samples = torch::rand({ 10, 5 });
        vector<string> features = { "feature1", "feature2", "feature3", "feature4", "feature5" };
        string className = "class1";
        int classNumStates = 2;

        bayesnet::Metrics obj(samples, features, className, classNumStates);

        REQUIRE(obj.getScoresKBest().size() == 0);
    }

    SECTION("Test SelectKBestWeighted")
    {
        torch::Tensor samples = torch::rand({ 10, 5 });
        vector<string> features = { "feature1", "feature2", "feature3", "feature4", "feature5" };
        string className = "class1";
        int classNumStates = 2;

        bayesnet::Metrics obj(samples, features, className, classNumStates);

        torch::Tensor weights = torch::ones({ 5 });

        vector<int> kBest = obj.SelectKBestWeighted(weights, true, 3);

        REQUIRE(kBest.size() == 3);
    }

    SECTION("Test mutualInformation")
    {
        torch::Tensor samples = torch::rand({ 10, 5 });
        vector<string> features = { "feature1", "feature2", "feature3", "feature4", "feature5" };
        string className = "class1";
        int classNumStates = 2;

        bayesnet::Metrics obj(samples, features, className, classNumStates);

        torch::Tensor firstFeature = samples.select(1, 0);
        torch::Tensor secondFeature = samples.select(1, 1);
        torch::Tensor weights = torch::ones({ 10 });

        double mi = obj.mutualInformation(firstFeature, secondFeature, weights);

        REQUIRE(mi >= 0);
    }
}