#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <string>
#include "TestUtils.h"
#include "bayesnet/classifiers/TAN.h"


TEST_CASE("Test Cannot build dataset with wrong data vector", "[Classifier]")
{
    auto model = bayesnet::TAN();
    auto raw = RawDatasets("iris", true);
    raw.yv.pop_back();
    REQUIRE_THROWS_AS(model.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv), std::runtime_error);
    REQUIRE_THROWS_WITH(model.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv), "* Error in X and y dimensions *\nX dimensions: [4, 150]\ny dimensions: [149]");
}
TEST_CASE("Test Cannot build dataset with wrong data tensor", "[Classifier]")
{
    auto model = bayesnet::TAN();
    auto raw = RawDatasets("iris", true);
    auto yshort = torch::zeros({ 149 }, torch::kInt32);
    REQUIRE_THROWS_AS(model.fit(raw.Xt, yshort, raw.featurest, raw.classNamet, raw.statest), std::runtime_error);
    REQUIRE_THROWS_WITH(model.fit(raw.Xt, yshort, raw.featurest, raw.classNamet, raw.statest), "* Error in X and y dimensions *\nX dimensions: [4, 150]\ny dimensions: [149]");
}