#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <string>
#include "TestUtils.h"
#include "Network.h"

TEST_CASE("Test Bayesian Network", "[BayesNet]")
{

    auto raw = RawDatasets("iris", true);

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
        REQUIRE(net.getNumEdges() == 2);
        net.addEdge("A", "C");
        REQUIRE(net.getEdges() == vector<pair<string, string>>{ {"A", "B"}, { "A", "C" }, { "B", "C" } });
        REQUIRE(net.getNumEdges() == 3);
    }
    SECTION("Test getNodes")
    {
        auto net = bayesnet::Network();
        net.addNode("A");
        net.addNode("B");
        auto& nodes = net.getNodes();
        REQUIRE(nodes.count("A") == 1);
        REQUIRE(nodes.count("B") == 1);
    }

    SECTION("Test fit")
    {
        auto net = bayesnet::Network();
        // net.fit(raw.Xv, raw.yv, raw.weightsv, raw.featuresv, raw.classNamev, raw.statesv);
        net.fit(raw.Xt, raw.yt, raw.weights, raw.featurest, raw.classNamet, raw.statest);
        REQUIRE(net.getClassName() == "class");
    }

    // SECTION("Test predict")
    // {
    //     auto net = bayesnet::Network();
    //     net.fit(raw.Xv, raw.yv, raw.weightsv, raw.featuresv, raw.classNamev, raw.statesv);
    //     vector<vector<int>> test = { {1, 2, 0, 1}, {0, 1, 2, 0}, {1, 1, 1, 1}, {0, 0, 0, 0}, {2, 2, 2, 2} };
    //     vector<int> y_test = { 0, 1, 1, 0, 2 };
    //     auto y_pred = net.predict(test);
    //     REQUIRE(y_pred == y_test);
    // }

    //     SECTION("Test predict_proba")
    //     {
    //         auto net = bayesnet::Network();
    //         net.fit(raw.Xv, raw.yv, raw.weightsv, raw.featuresv, raw.classNamev, raw.statesv);
    //         vector<vector<int>> test = { {1, 2, 0, 1}, {0, 1, 2, 0}, {1, 1, 1, 1}, {0, 0, 0, 0}, {2, 2, 2, 2} };
    //         auto y_test = { 0, 1, 1, 0, 2 };
    //         auto y_pred = net.predict(test);
    //         REQUIRE(y_pred == y_test);
    //     }
}

// SECTION("Test score")
// {
//     auto net = bayesnet::Network();
//     net.fit(Xd, y, weights, features, className, states);
//     auto test = { {1, 2, 0, 1}, {0, 1, 2, 0}, {1, 1, 1, 1}, {0, 0, 0, 0}, {2, 2, 2, 2} };
//     auto score = net.score(X, y);
//     REQUIRE(score == Catch::Approx();
// }

// SECTION("Test topological_sort")
// {
//     auto net = bayesnet::Network();
//     net.addNode("A");
//     net.addNode("B");
//     net.addNode("C");
//     net.addEdge("A", "B");
//     net.addEdge("A", "C");
//     auto sorted = net.topological_sort();
//     REQUIRE(sorted.size() == 3);
//     REQUIRE(sorted[0] == "A");
//     REQUIRE((sorted[1] == "B" && sorted[2] == "C") || (sorted[1] == "C" && sorted[2] == "B"));
// }

// SECTION("Test show")
// {
//     auto net = bayesnet::Network();
//     net.addNode("A");
//     net.addNode("B");
//     net.addNode("C");
//     net.addEdge("A", "B");
//     net.addEdge("A", "C");
//     auto str = net.show();
//     REQUIRE(str.size() == 3);
//     REQUIRE(str[0] == "A");
//     REQUIRE(str[1] == "B -> C");
//     REQUIRE(str[2] == "C");
// }

// SECTION("Test graph")
// {
//     auto net = bayesnet::Network();
//     net.addNode("A");
//     net.addNode("B");
//     net.addNode("C");
//     net.addEdge("A", "B");
//     net.addEdge("A", "C");
//     auto str = net.graph("Test Graph");
//     REQUIRE(str.size() == 6);
//     REQUIRE(str[0] == "digraph \"Test Graph\" {");
//     REQUIRE(str[1] == "  A -> B;");
//     REQUIRE(str[2] == "  A -> C;");
//     REQUIRE(str[3] == "  B [shape=ellipse];");
//     REQUIRE(str[4] == "  C [shape=ellipse];");
//     REQUIRE(str[5] == "}");
// }

// SECTION("Test initialize")
// {
//     auto net = bayesnet::Network();
//     net.addNode("A");
//     net.addNode("B");
//     net.addNode("C");
//     net.addEdge("A", "B");
//     net.addEdge("A", "C");
//     net.initialize();
//     REQUIRE(net.getNodes().size() == 0);
//     REQUIRE(net.getEdges().size() == 0);
//     REQUIRE(net.getFeatures().size() == 0);
//     REQUIRE(net.getClassNumStates() == 0);
//     REQUIRE(net.getClassName().empty());
//     REQUIRE(net.getStates() == 0);
//     REQUIRE(net.getSamples().numel() == 0);
// }

// SECTION("Test dump_cpt")
// {
//     auto net = bayesnet::Network();
//     net.addNode("A");
//     net.addNode("B");
//     net.addNode("C");
//     net.addEdge("A", "B");
//     net.addEdge("A", "C");
//     net.setClassName("C");
//     net.setStates({ {"A", {0, 1}}, {"B", {0, 1}}, {"C", {0, 1, 2}} });
//     net.fit({ {0, 0}, {0, 1}, {1, 0}, {1, 1} }, { 0, 1, 1, 2 }, {}, { "A", "B" }, "C", { {"A", {0, 1}}, {"B", {0, 1}}, {"C", {0, 1, 2}} });
//     net.dump_cpt();
//     // TODO: Check that the file was created and contains the expected data
// }

// SECTION("Test version")
// {
//     auto net = bayesnet::Network();
//     REQUIRE(net.version() == "0.2.0");
// }
// }

// }
