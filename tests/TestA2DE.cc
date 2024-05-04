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
    std::cout << "Score A2DE: " << clf.score(raw.Xv, raw.yv) << std::endl;
    // REQUIRE(clf.getNumberOfNodes() == 90);
    // REQUIRE(clf.getNumberOfEdges() == 153);
    // REQUIRE(clf.getNotes().size() == 2);
    // REQUIRE(clf.getNotes()[0] == "Used features in initialization: 6 of 9 with CFS");
    // REQUIRE(clf.getNotes()[1] == "Number of models: 9");
}
