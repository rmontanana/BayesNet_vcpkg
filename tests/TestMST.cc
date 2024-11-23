// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <string>
#include <vector>
#include "TestUtils.h"
#include "bayesnet/utils/Mst.h"


TEST_CASE("MST::insertElement tests", "[MST]")
{
    bayesnet::MST mst({}, torch::tensor({}), 0);
    SECTION("Insert into an empty list")
    {
        std::list<int> variables;
        mst.insertElement(variables, 5);
        REQUIRE(variables == std::list<int>{5});
    }
    SECTION("Insert a non-duplicate element")
    {
        std::list<int> variables = { 1, 2, 3 };
        mst.insertElement(variables, 4);
        REQUIRE(variables == std::list<int>{4, 1, 2, 3});
    }
    SECTION("Insert a duplicate element")
    {
        std::list<int> variables = { 1, 2, 3 };
        mst.insertElement(variables, 2);
        REQUIRE(variables == std::list<int>{1, 2, 3});
    }
}

TEST_CASE("MST::reorder tests", "[MST]")
{
    bayesnet::MST mst({}, torch::tensor({}), 0);
    SECTION("Reorder simple graph")
    {
        std::vector<std::pair<float, std::pair<int, int>>> T = { {2.0, {1, 2}}, {1.0, {0, 1}} };
        auto result = mst.reorder(T, 0);
        REQUIRE(result == std::vector<std::pair<int, int>>{{0, 1}, { 1, 2 }});
    }
    SECTION("Reorder with disconnected graph")
    {
        std::vector<std::pair<float, std::pair<int, int>>> T = { {2.0, {1, 2}}, {1.0, {0, 1}} };
        auto result = mst.reorder(T, 0);
        REQUIRE(result == std::vector<std::pair<int, int>>{{0, 1}, { 2, 3 }});
    }
}

TEST_CASE("MST::maximumSpanningTree tests", "[MST]")
{
    std::vector<std::string> features = { "A", "B", "C" };
    auto weights = torch::tensor({
        {0.0, 1.0, 2.0},
        {1.0, 0.0, 3.0},
        {2.0, 3.0, 0.0}
        });
    bayesnet::MST mst(features, weights, 0);

    SECTION("MST of a complete graph")
    {
        auto result = mst.maximumSpanningTree();
        REQUIRE(result.size() == 2); // Un MST para 3 nodos tiene 2 aristas
    }
}