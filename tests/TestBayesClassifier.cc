// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <string>
#include "TestUtils.h"
#include "bayesnet/classifiers/TAN.h"
#include "bayesnet/classifiers/KDB.h"
#include "bayesnet/classifiers/KDBLd.h"


TEST_CASE("Test Cannot build dataset with wrong data vector", "[Classifier]")
{
    auto model = bayesnet::TAN();
    auto raw = RawDatasets("iris", true);
    raw.yv.pop_back();
    REQUIRE_THROWS_AS(model.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing), std::runtime_error);
    REQUIRE_THROWS_WITH(model.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing), "* Error in X and y dimensions *\nX dimensions: [4, 150]\ny dimensions: [149]");
}
TEST_CASE("Test Cannot build dataset with wrong data tensor", "[Classifier]")
{
    auto model = bayesnet::TAN();
    auto raw = RawDatasets("iris", true);
    auto yshort = torch::zeros({ 149 }, torch::kInt32);
    REQUIRE_THROWS_AS(model.fit(raw.Xt, yshort, raw.features, raw.className, raw.states, raw.smoothing), std::runtime_error);
    REQUIRE_THROWS_WITH(model.fit(raw.Xt, yshort, raw.features, raw.className, raw.states, raw.smoothing), "* Error in X and y dimensions *\nX dimensions: [4, 150]\ny dimensions: [149]");
}
TEST_CASE("Invalid data type", "[Classifier]")
{
    auto model = bayesnet::TAN();
    auto raw = RawDatasets("iris", false);
    REQUIRE_THROWS_AS(model.fit(raw.Xt, raw.yt, raw.features, raw.className, raw.states, raw.smoothing), std::invalid_argument);
    REQUIRE_THROWS_WITH(model.fit(raw.Xt, raw.yt, raw.features, raw.className, raw.states, raw.smoothing), "dataset (X, y) must be of type Integer");
}
TEST_CASE("Invalid number of features", "[Classifier]")
{
    auto model = bayesnet::TAN();
    auto raw = RawDatasets("iris", true);
    auto Xt = torch::cat({ raw.Xt, torch::zeros({ 1, 150 }, torch::kInt32) }, 0);
    REQUIRE_THROWS_AS(model.fit(Xt, raw.yt, raw.features, raw.className, raw.states, raw.smoothing), std::invalid_argument);
    REQUIRE_THROWS_WITH(model.fit(Xt, raw.yt, raw.features, raw.className, raw.states, raw.smoothing), "Classifier: X 5 and features 4 must have the same number of features");
}
TEST_CASE("Invalid class name", "[Classifier]")
{
    auto model = bayesnet::TAN();
    auto raw = RawDatasets("iris", true);
    REQUIRE_THROWS_AS(model.fit(raw.Xt, raw.yt, raw.features, "duck", raw.states, raw.smoothing), std::invalid_argument);
    REQUIRE_THROWS_WITH(model.fit(raw.Xt, raw.yt, raw.features, "duck", raw.states, raw.smoothing), "class name not found in states");
}
TEST_CASE("Invalid feature name", "[Classifier]")
{
    auto model = bayesnet::TAN();
    auto raw = RawDatasets("iris", true);
    auto statest = raw.states;
    statest.erase("petallength");
    REQUIRE_THROWS_AS(model.fit(raw.Xt, raw.yt, raw.features, raw.className, statest, raw.smoothing), std::invalid_argument);
    REQUIRE_THROWS_WITH(model.fit(raw.Xt, raw.yt, raw.features, raw.className, statest, raw.smoothing), "feature [petallength] not found in states");
}
TEST_CASE("Invalid hyperparameter", "[Classifier]")
{
    auto model = bayesnet::KDB(2);
    auto raw = RawDatasets("iris", true);
    REQUIRE_THROWS_AS(model.setHyperparameters({ { "alpha", "0.0" } }), std::invalid_argument);
    REQUIRE_THROWS_WITH(model.setHyperparameters({ { "alpha", "0.0" } }), "Invalid hyperparameters{\"alpha\":\"0.0\"}");
}
TEST_CASE("Topological order", "[Classifier]")
{
    auto model = bayesnet::TAN();
    auto raw = RawDatasets("iris", true);
    model.fit(raw.Xt, raw.yt, raw.features, raw.className, raw.states, raw.smoothing);
    auto order = model.topological_order();
    REQUIRE(order.size() == 4);
    REQUIRE(order[0] == "petallength");
    REQUIRE(order[1] == "sepallength");
    REQUIRE(order[2] == "sepalwidth");
    REQUIRE(order[3] == "petalwidth");
}
TEST_CASE("Dump_cpt", "[Classifier]")
{
    auto model = bayesnet::TAN();
    auto raw = RawDatasets("iris", true);
    model.fit(raw.Xt, raw.yt, raw.features, raw.className, raw.states, raw.smoothing);
    auto cpt = model.dump_cpt();
    REQUIRE(cpt.size() == 1718);
}
TEST_CASE("Not fitted model", "[Classifier]")
{
    auto model = bayesnet::TAN();
    auto raw = RawDatasets("iris", true);
    auto message = "Classifier has not been fitted";
    // tensors
    REQUIRE_THROWS_AS(model.predict(raw.Xt), std::logic_error);
    REQUIRE_THROWS_WITH(model.predict(raw.Xt), message);
    REQUIRE_THROWS_AS(model.predict_proba(raw.Xt), std::logic_error);
    REQUIRE_THROWS_WITH(model.predict_proba(raw.Xt), message);
    REQUIRE_THROWS_AS(model.score(raw.Xt, raw.yt), std::logic_error);
    REQUIRE_THROWS_WITH(model.score(raw.Xt, raw.yt), message);
    // vectors
    REQUIRE_THROWS_AS(model.predict(raw.Xv), std::logic_error);
    REQUIRE_THROWS_WITH(model.predict(raw.Xv), message);
    REQUIRE_THROWS_AS(model.predict_proba(raw.Xv), std::logic_error);
    REQUIRE_THROWS_WITH(model.predict_proba(raw.Xv), message);
    REQUIRE_THROWS_AS(model.score(raw.Xv, raw.yv), std::logic_error);
    REQUIRE_THROWS_WITH(model.score(raw.Xv, raw.yv), message);
}
TEST_CASE("KDB Graph", "[Classifier]")
{
    auto model = bayesnet::KDB(2);
    auto raw = RawDatasets("iris", true);
    model.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    auto graph = model.graph();
    REQUIRE(graph.size() == 15);
}
TEST_CASE("KDBLd Graph", "[Classifier]")
{
    auto model = bayesnet::KDBLd(2);
    auto raw = RawDatasets("iris", false);
    model.fit(raw.Xt, raw.yt, raw.features, raw.className, raw.states, raw.smoothing);
    auto graph = model.graph();
    REQUIRE(graph.size() == 15);
}