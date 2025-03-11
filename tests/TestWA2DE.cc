// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <type_traits>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "bayesnet/ensembles/WA2DE.h"
#include "TestUtils.h"


TEST_CASE("Fit and Score", "[WA2DE]")
{
    auto raw = RawDatasets("iris", true);
    auto clf = bayesnet::WA2DE();
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    REQUIRE(clf.score(raw.Xt, raw.yt) == Catch::Approx(0.6333333333333333).epsilon(raw.epsilon));
}
TEST_CASE("Test graph", "[WA2DE]")
{
    auto raw = RawDatasets("iris", true);
    auto clf = bayesnet::WA2DE();
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    auto graph = clf.graph("BayesNet WA2DE");
    REQUIRE(graph.size() == 2);
    REQUIRE(graph[0] == "BayesNet WA2DE");
    REQUIRE(graph[1] == "Graph visualization not implemented.");
}
