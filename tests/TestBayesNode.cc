// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <string>
#include "TestUtils.h"
#include "bayesnet/network/Network.h"



TEST_CASE("Test Node children and parents", "[Node]")
{
    auto node = bayesnet::Node("Node");
    REQUIRE(node.getName() == "Node");
    auto parent_1 = bayesnet::Node("P1");
    auto parent_2 = bayesnet::Node("P2");
    auto child_1 = bayesnet::Node("H1");
    auto child_2 = bayesnet::Node("H2");
    auto child_3 = bayesnet::Node("H3");
    node.addParent(&parent_1);
    node.addParent(&parent_2);
    node.addChild(&child_1);
    node.addChild(&child_2);
    node.addChild(&child_3);
    auto parents = node.getParents();
    auto children = node.getChildren();
    REQUIRE(parents.size() == 2);
    REQUIRE(children.size() == 3);
    REQUIRE(parents[0]->getName() == "P1");
    REQUIRE(parents[1]->getName() == "P2");
    REQUIRE(children[0]->getName() == "H1");
    REQUIRE(children[1]->getName() == "H2");
    REQUIRE(children[2]->getName() == "H3");
    node.removeParent(&parent_1);
    node.removeChild(&child_1);
    parents = node.getParents();
    children = node.getChildren();
    REQUIRE(parents.size() == 1);
    REQUIRE(children.size() == 2);
    node.clear();
    parents = node.getParents();
    children = node.getChildren();
    REQUIRE(parents.size() == 0);
    REQUIRE(children.size() == 0);
}
TEST_CASE("TEST MinFill method", "[Node]")
{
    // Generate a test to test the minFill method of the Node class
    // Create a graph with 5 nodes
    // The graph is a chain with some additional edges
    // 0 -> 1,2,3
    // 1 -> 2,4
    // 2 -> 3
    // 3 -> 4
    auto node_0 = bayesnet::Node("0");
    auto node_1 = bayesnet::Node("1");
    auto node_2 = bayesnet::Node("2");
    auto node_3 = bayesnet::Node("3");
    auto node_4 = bayesnet::Node("4");
    // node 0
    node_0.addChild(&node_1);
    node_0.addChild(&node_2);
    node_0.addChild(&node_3);
    // node 1
    node_1.addChild(&node_2);
    node_1.addChild(&node_4);
    node_1.addParent(&node_0);
    // node 2
    node_2.addChild(&node_3);
    node_2.addChild(&node_4);
    node_2.addParent(&node_0);
    node_2.addParent(&node_1);
    // node 3
    node_3.addChild(&node_4);
    node_3.addParent(&node_0);
    node_3.addParent(&node_2);
    // node 4
    node_4.addParent(&node_1);
    node_4.addParent(&node_3);
    REQUIRE(node_0.minFill() == 3);
    REQUIRE(node_1.minFill() == 3);
    REQUIRE(node_2.minFill() == 6);
    REQUIRE(node_3.minFill() == 3);
    REQUIRE(node_4.minFill() == 1);
}