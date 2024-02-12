#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <vector>
#include <map>
#include <string>
#include "KDB.h"
#include "TAN.h"
#include "SPODE.h"
#include "AODE.h"
#include "BoostAODE.h"
#include "TANLd.h"
#include "KDBLd.h"
#include "SPODELd.h"
#include "AODELd.h"
#include "TestUtils.h"

TEST_CASE("Library check version", "[BayesNet]")
{
    auto clf = bayesnet::KDB(2);
    REQUIRE(clf.getVersion() == "1.0.1");
}
TEST_CASE("Test Bayesian Classifiers score", "[BayesNet]")
{
    map <pair<std::string, std::string>, float> scores = {
        // Diabetes
        {{"diabetes", "AODE"}, 0.811198}, {{"diabetes", "KDB"}, 0.852865}, {{"diabetes", "SPODE"}, 0.802083}, {{"diabetes", "TAN"}, 0.821615},
        {{"diabetes", "AODELd"}, 0.8138f}, {{"diabetes", "KDBLd"}, 0.80208f}, {{"diabetes", "SPODELd"}, 0.78646f}, {{"diabetes", "TANLd"}, 0.8099f},  {{"diabetes", "BoostAODE"}, 0.83984f},
        // Ecoli
        {{"ecoli", "AODE"}, 0.889881}, {{"ecoli", "KDB"}, 0.889881}, {{"ecoli", "SPODE"}, 0.880952}, {{"ecoli", "TAN"}, 0.892857},
        {{"ecoli", "AODELd"}, 0.8869f}, {{"ecoli", "KDBLd"}, 0.875f}, {{"ecoli", "SPODELd"}, 0.84226f}, {{"ecoli", "TANLd"}, 0.86905f}, {{"ecoli", "BoostAODE"}, 0.89583f},
        // Glass
        {{"glass", "AODE"}, 0.78972}, {{"glass", "KDB"}, 0.827103}, {{"glass", "SPODE"}, 0.775701}, {{"glass", "TAN"}, 0.827103},
        {{"glass", "AODELd"}, 0.79439f}, {{"glass", "KDBLd"}, 0.85047f}, {{"glass", "SPODELd"}, 0.79439f}, {{"glass", "TANLd"}, 0.86449f}, {{"glass", "BoostAODE"}, 0.84579f},
        // Iris
        {{"iris", "AODE"}, 0.973333}, {{"iris", "KDB"}, 0.973333}, {{"iris", "SPODE"}, 0.973333}, {{"iris", "TAN"}, 0.973333},
        {{"iris", "AODELd"}, 0.973333}, {{"iris", "KDBLd"}, 0.973333}, {{"iris", "SPODELd"}, 0.96f}, {{"iris", "TANLd"}, 0.97333f}, {{"iris", "BoostAODE"}, 0.98f}
    };

    std::string file_name = GENERATE("glass", "iris", "ecoli", "diabetes");
    auto raw = RawDatasets(file_name, false);

    SECTION("Test TAN classifier (" + file_name + ")")
    {
        auto clf = bayesnet::TAN();
        clf.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
        auto score = clf.score(raw.Xv, raw.yv);
        //scores[{file_name, "TAN"}] = score;
        REQUIRE(score == Catch::Approx(scores[{file_name, "TAN"}]).epsilon(raw.epsilon));
    }
    SECTION("Test TANLd classifier (" + file_name + ")")
    {
        auto clf = bayesnet::TANLd();
        clf.fit(raw.Xt, raw.yt, raw.featurest, raw.classNamet, raw.statest);
        auto score = clf.score(raw.Xt, raw.yt);
        //scores[{file_name, "TANLd"}] = score;
        REQUIRE(score == Catch::Approx(scores[{file_name, "TANLd"}]).epsilon(raw.epsilon));
    }
    SECTION("Test KDB classifier (" + file_name + ")")
    {
        auto clf = bayesnet::KDB(2);
        clf.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
        auto score = clf.score(raw.Xv, raw.yv);
        //scores[{file_name, "KDB"}] = score;
        REQUIRE(score == Catch::Approx(scores[{file_name, "KDB"
        }]).epsilon(raw.epsilon));
    }
    SECTION("Test KDBLd classifier (" + file_name + ")")
    {
        auto clf = bayesnet::KDBLd(2);
        clf.fit(raw.Xt, raw.yt, raw.featurest, raw.classNamet, raw.statest);
        auto score = clf.score(raw.Xt, raw.yt);
        //scores[{file_name, "KDBLd"}] = score;
        REQUIRE(score == Catch::Approx(scores[{file_name, "KDBLd"
        }]).epsilon(raw.epsilon));
    }
    SECTION("Test SPODE classifier (" + file_name + ")")
    {
        auto clf = bayesnet::SPODE(1);
        clf.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
        auto score = clf.score(raw.Xv, raw.yv);
        // scores[{file_name, "SPODE"}] = score;
        REQUIRE(score == Catch::Approx(scores[{file_name, "SPODE"}]).epsilon(raw.epsilon));
    }
    SECTION("Test SPODELd classifier (" + file_name + ")")
    {
        auto clf = bayesnet::SPODELd(1);
        clf.fit(raw.Xt, raw.yt, raw.featurest, raw.classNamet, raw.statest);
        auto score = clf.score(raw.Xt, raw.yt);
        // scores[{file_name, "SPODELd"}] = score;
        REQUIRE(score == Catch::Approx(scores[{file_name, "SPODELd"}]).epsilon(raw.epsilon));
    }
    SECTION("Test AODE classifier (" + file_name + ")")
    {
        auto clf = bayesnet::AODE();
        clf.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
        auto score = clf.score(raw.Xv, raw.yv);
        // scores[{file_name, "AODE"}] = score;
        REQUIRE(score == Catch::Approx(scores[{file_name, "AODE"}]).epsilon(raw.epsilon));
    }
    SECTION("Test AODELd classifier (" + file_name + ")")
    {
        auto clf = bayesnet::AODELd();
        clf.fit(raw.Xt, raw.yt, raw.featurest, raw.classNamet, raw.statest);
        auto score = clf.score(raw.Xt, raw.yt);
        // scores[{file_name, "AODELd"}] = score;
        REQUIRE(score == Catch::Approx(scores[{file_name, "AODELd"}]).epsilon(raw.epsilon));
    }
    SECTION("Test BoostAODE classifier (" + file_name + ")")
    {
        auto clf = bayesnet::BoostAODE();
        clf.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
        auto score = clf.score(raw.Xv, raw.yv);
        // scores[{file_name, "BoostAODE"}] = score;
        REQUIRE(score == Catch::Approx(scores[{file_name, "BoostAODE"}]).epsilon(raw.epsilon));
    }
    // for (auto scores : scores) {
    //     std::cout << "{{\"" << scores.first.first << "\", \"" << scores.first.second << "\"}, " << scores.second << "}, ";
    // }
}
TEST_CASE("Models features", "[BayesNet]")
{
    auto graph = std::vector<std::string>({ "digraph BayesNet {\nlabel=<BayesNet Test>\nfontsize=30\nfontcolor=blue\nlabelloc=t\nlayout=circo\n",
        "class [shape=circle, fontcolor=red, fillcolor=lightblue, style=filled ] \n",
        "class -> sepallength", "class -> sepalwidth", "class -> petallength", "class -> petalwidth", "petallength [shape=circle] \n",
        "petallength -> sepallength", "petalwidth [shape=circle] \n", "sepallength [shape=circle] \n",
        "sepallength -> sepalwidth", "sepalwidth [shape=circle] \n", "sepalwidth -> petalwidth", "}\n"
        }
    );
    auto raw = RawDatasets("iris", true);
    auto clf = bayesnet::TAN();
    clf.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
    REQUIRE(clf.getNumberOfNodes() == 5);
    REQUIRE(clf.getNumberOfEdges() == 7);
    REQUIRE(clf.show() == std::vector<std::string>{"class -> sepallength, sepalwidth, petallength, petalwidth, ", "petallength -> sepallength, ", "petalwidth -> ", "sepallength -> sepalwidth, ", "sepalwidth -> petalwidth, "});
    REQUIRE(clf.graph("Test") == graph);
}
TEST_CASE("Get num features & num edges", "[BayesNet]")
{
    auto raw = RawDatasets("iris", true);
    auto clf = bayesnet::KDB(2);
    clf.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
    REQUIRE(clf.getNumberOfNodes() == 5);
    REQUIRE(clf.getNumberOfEdges() == 8);
}
TEST_CASE("BoostAODE feature_select CFS", "[BayesNet]")
{
    auto raw = RawDatasets("glass", true);
    auto clf = bayesnet::BoostAODE();
    clf.setHyperparameters({ {"select_features", "CFS"} });
    clf.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
    REQUIRE(clf.getNumberOfNodes() == 90);
    REQUIRE(clf.getNumberOfEdges() == 153);
    REQUIRE(clf.getNotes().size() == 2);
    REQUIRE(clf.getNotes()[0] == "Used features in initialization: 6 of 9 with CFS");
    REQUIRE(clf.getNotes()[1] == "Number of models: 9");
}
TEST_CASE("BoostAODE test used features in train note", "[BayesNet]")
{
    auto raw = RawDatasets("diabetes", true);
    auto clf = bayesnet::BoostAODE();
    clf.setHyperparameters({
        {"ascending",true},
        {"convergence", true},
        {"repeatSparent",true},
        {"select_features","CFS"}
        });
    clf.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
    REQUIRE(clf.getNumberOfNodes() == 72);
    REQUIRE(clf.getNumberOfEdges() == 120);
    REQUIRE(clf.getNotes().size() == 3);
    REQUIRE(clf.getNotes()[0] == "Used features in initialization: 6 of 8 with CFS");
    REQUIRE(clf.getNotes()[1] == "Used features in train: 7 of 8");
    REQUIRE(clf.getNotes()[2] == "Number of models: 8");
}
