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

TEST_CASE("Test Bayesian Classifiers score", "[BayesNet]")
{
    map <pair<string, string>, float> scores = {
        // Diabetes
        {{"diabetes", "AODE"}, 0.811198}, {{"diabetes", "KDB"}, 0.852865}, {{"diabetes", "SPODE"}, 0.802083}, {{"diabetes", "TAN"}, 0.821615},
        {{"diabetes", "AODELd"}, 0.811198}, {{"diabetes", "KDBLd"}, 0.852865}, {{"diabetes", "SPODELd"}, 0.802083}, {{"diabetes", "TANLd"}, 0.821615},  {{"diabetes", "BoostAODE"}, 0.821615},
        // Ecoli
        {{"ecoli", "AODE"}, 0.889881}, {{"ecoli", "KDB"}, 0.889881}, {{"ecoli", "SPODE"}, 0.880952}, {{"ecoli", "TAN"}, 0.892857},
        {{"ecoli", "AODELd"}, 0.889881}, {{"ecoli", "KDBLd"}, 0.889881}, {{"ecoli", "SPODELd"}, 0.880952}, {{"ecoli", "TANLd"}, 0.892857}, {{"ecoli", "BoostAODE"}, 0.892857},
        // Glass
        {{"glass", "AODE"}, 0.78972}, {{"glass", "KDB"}, 0.827103}, {{"glass", "SPODE"}, 0.775701}, {{"glass", "TAN"}, 0.827103},
        {{"glass", "AODELd"}, 0.78972}, {{"glass", "KDBLd"}, 0.827103}, {{"glass", "SPODELd"}, 0.775701}, {{"glass", "TANLd"}, 0.827103}, {{"glass", "BoostAODE"}, 0.827103},
        // Iris
        {{"iris", "AODE"}, 0.973333}, {{"iris", "KDB"}, 0.973333}, {{"iris", "SPODE"}, 0.973333}, {{"iris", "TAN"}, 0.973333},
        {{"iris", "AODELd"}, 0.973333}, {{"iris", "KDBLd"}, 0.973333}, {{"iris", "SPODELd"}, 0.973333}, {{"iris", "TANLd"}, 0.973333}, {{"iris", "BoostAODE"}, 0.973333}
    };

    string file_name = GENERATE("glass", "iris", "ecoli", "diabetes");
    auto [XCont, yCont, featuresCont, classNameCont, statesCont] = loadDataset(file_name, true, false);
    auto [XDisc, yDisc, featuresDisc, className, statesDisc] = loadFile(file_name);

    SECTION("Test TAN classifier (" + file_name + ")")
    {
        auto clf = bayesnet::TAN();
        clf.fit(XDisc, yDisc, featuresDisc, className, statesDisc);
        auto score = clf.score(XDisc, yDisc);
        //scores[{file_name, "TAN"}] = score;
        REQUIRE(score == Catch::Approx(scores[{file_name, "TAN"}]).epsilon(1e-6));
    }
    SECTION("Test TANLd classifier (" + file_name + ")")
    {
        auto clf = bayesnet::TANLd();
        clf.fit(XCont, yCont, featuresCont, classNameCont, statesCont);
        auto score = clf.score(XCont, yCont);
        //scores[{file_name, "TANLd"}] = score;
        REQUIRE(score == Catch::Approx(scores[{file_name, "TANLd"}]).epsilon(1e-6));
    }
    SECTION("Test KDB classifier (" + file_name + ")")
    {
        auto clf = bayesnet::KDB(2);
        clf.fit(XDisc, yDisc, featuresDisc, className, statesDisc);
        auto score = clf.score(XDisc, yDisc);
        //scores[{file_name, "KDB"}] = score;
        REQUIRE(score == Catch::Approx(scores[{file_name, "KDB"
        }]).epsilon(1e-6));
    }
    SECTION("Test KDBLd classifier (" + file_name + ")")
    {
        auto clf = bayesnet::KDBLd(2);
        clf.fit(XCont, yCont, featuresCont, classNameCont, statesCont);
        auto score = clf.score(XCont, yCont);
        //scores[{file_name, "KDBLd"}] = score;
        REQUIRE(score == Catch::Approx(scores[{file_name, "KDBLd"
        }]).epsilon(1e-6));
    }
    SECTION("Test SPODE classifier (" + file_name + ")")
    {
        auto clf = bayesnet::SPODE(1);
        clf.fit(XDisc, yDisc, featuresDisc, className, statesDisc);
        auto score = clf.score(XDisc, yDisc);
        // scores[{file_name, "SPODE"}] = score;
        REQUIRE(score == Catch::Approx(scores[{file_name, "SPODE"}]).epsilon(1e-6));
    }
    SECTION("Test SPODELd classifier (" + file_name + ")")
    {
        auto clf = bayesnet::SPODELd(1);
        clf.fit(XCont, yCont, featuresCont, classNameCont, statesCont);
        auto score = clf.score(XCont, yCont);
        // scores[{file_name, "SPODELd"}] = score;
        REQUIRE(score == Catch::Approx(scores[{file_name, "SPODELd"}]).epsilon(1e-6));
    }
    SECTION("Test AODE classifier (" + file_name + ")")
    {
        auto clf = bayesnet::AODE();
        clf.fit(XDisc, yDisc, featuresDisc, className, statesDisc);
        auto score = clf.score(XDisc, yDisc);
        // scores[{file_name, "AODE"}] = score;
        REQUIRE(score == Catch::Approx(scores[{file_name, "AODE"}]).epsilon(1e-6));
    }
    SECTION("Test AODELd classifier (" + file_name + ")")
    {
        auto clf = bayesnet::AODE();
        clf.fit(XCont, yCont, featuresCont, classNameCont, statesCont);
        auto score = clf.score(XCont, yCont);
        // scores[{file_name, "AODELd"}] = score;
        REQUIRE(score == Catch::Approx(scores[{file_name, "AODELd"}]).epsilon(1e-6));
    }
    SECTION("Test BoostAODE classifier (" + file_name + ")")
    {
        auto clf = bayesnet::BoostAODE();
        clf.fit(XDisc, yDisc, featuresDisc, className, statesDisc);
        auto score = clf.score(XDisc, yDisc);
        // scores[{file_name, "BoostAODE"}] = score;
        REQUIRE(score == Catch::Approx(scores[{file_name, "BoostAODE"}]).epsilon(1e-6));
    }
    // for (auto scores : scores) {
    //     cout << "{{\"" << scores.first.first << "\", \"" << scores.first.second << "\"}, " << scores.second << "}, ";
    // }
}
TEST_CASE("Models featuresDisc")
{
    auto graph = vector<string>({ "digraph BayesNet {\nlabel=<BayesNet Test>\nfontsize=30\nfontcolor=blue\nlabelloc=t\nlayout=circo\n",
        "class [shape=circle, fontcolor=red, fillcolor=lightblue, style=filled ] \n",
        "class -> sepallength", "class -> sepalwidth", "class -> petallength", "class -> petalwidth", "petallength [shape=circle] \n",
        "petallength -> sepallength", "petalwidth [shape=circle] \n", "sepallength [shape=circle] \n",
        "sepallength -> sepalwidth", "sepalwidth [shape=circle] \n", "sepalwidth -> petalwidth", "}\n"
        }
    );

    auto clf = bayesnet::TAN();
    auto [XDisc, yDisc, featuresDisc, className, statesDisc] = loadFile("iris");
    clf.fit(XDisc, yDisc, featuresDisc, className, statesDisc);
    REQUIRE(clf.getNumberOfNodes() == 5);
    REQUIRE(clf.getNumberOfEdges() == 7);
    REQUIRE(clf.show() == vector<string>{"class -> sepallength, sepalwidth, petallength, petalwidth, ", "petallength -> sepallength, ", "petalwidth -> ", "sepallength -> sepalwidth, ", "sepalwidth -> petalwidth, "});
    REQUIRE(clf.graph("Test") == graph);
}
TEST_CASE("Get num featuresDisc & num edges")
{
    auto [XDisc, yDisc, featuresDisc, className, statesDisc] = loadFile("iris");
    auto clf = bayesnet::KDB(2);
    clf.fit(XDisc, yDisc, featuresDisc, className, statesDisc);
    REQUIRE(clf.getNumberOfNodes() == 5);
    REQUIRE(clf.getNumberOfEdges() == 8);
}