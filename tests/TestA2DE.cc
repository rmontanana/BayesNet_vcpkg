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
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states);
    REQUIRE(clf.score(raw.Xv, raw.yv) == Catch::Approx(0.831776).epsilon(raw.epsilon));
    REQUIRE(clf.getNumberOfNodes() == 360);
    REQUIRE(clf.getNumberOfEdges() == 756);
    REQUIRE(clf.getNotes().size() == 0);
}
