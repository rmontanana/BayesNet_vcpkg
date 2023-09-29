#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <string>
#include "KDB.h"

TEST_CASE("Test Bayesian Network")
{
    auto [Xd, y, features, className, states] = loadFile("iris");

    SECTION("Test get features")
    {
        auto net = bayesnet::Network();
        net.addNode("A");
        net.addNode("B");
        REQUIRE(net.getFeatures() == vector<string>{"A", "B"});
        net.addNode("C");
        REQUIRE(net.getFeatures() == vector<string>{"A", "B", "C"});
    }
    SECTION("Test get edges")
    {
        auto net = bayesnet::Network();
        net.addNode("A");
        net.addNode("B");
        net.addNode("C");
        net.addEdge("A", "B");
        net.addEdge("B", "C");
        REQUIRE(net.getEdges() == vector<pair<string, string>>{ {"A", "B"}, { "B", "C" } });
        net.addEdge("A", "C");
        REQUIRE(net.getEdges() == vector<pair<string, string>>{ {"A", "B"}, { "A", "C" }, { "B", "C" } });
    }
}
