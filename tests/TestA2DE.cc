// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <type_traits>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "bayesnet/ensembles/A2DE.h"
#include "TestUtils.h"


TEST_CASE("Fit and Score", "[A2DE]")
{
    auto raw = RawDatasets("glass", true);
    auto clf = bayesnet::A2DE();
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    REQUIRE(clf.score(raw.Xv, raw.yv) == Catch::Approx(0.831776).epsilon(raw.epsilon));
    REQUIRE(clf.getNumberOfNodes() == 360);
    REQUIRE(clf.getNumberOfEdges() == 756);
    REQUIRE(clf.getNotes().size() == 0);
}
TEST_CASE("Test score with predict_voting", "[A2DE]")
{
    auto raw = RawDatasets("glass", true);
    auto clf = bayesnet::A2DE(true);
    auto hyperparameters = nlohmann::json{
       {"predict_voting", true},
    };
    clf.setHyperparameters(hyperparameters);
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    REQUIRE(clf.score(raw.Xv, raw.yv) == Catch::Approx(0.82243).epsilon(raw.epsilon));
    hyperparameters["predict_voting"] = false;
    clf.setHyperparameters(hyperparameters);
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    REQUIRE(clf.score(raw.Xv, raw.yv) == Catch::Approx(0.83178).epsilon(raw.epsilon));
}
TEST_CASE("Test graph", "[A2DE]")
{
    auto raw = RawDatasets("iris", true);
    auto clf = bayesnet::A2DE();
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    auto graph = clf.graph();
    REQUIRE(graph.size() == 78);
    REQUIRE(graph[0] == "digraph BayesNet {\nlabel=<BayesNet A2DE_0>\nfontsize=30\nfontcolor=blue\nlabelloc=t\nlayout=circo\n");
    REQUIRE(graph[1] == "class [shape=circle, fontcolor=red, fillcolor=lightblue, style=filled ] \n");
}
