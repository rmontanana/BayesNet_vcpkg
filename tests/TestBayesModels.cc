#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
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

const std::string ACTUAL_VERSION = "1.0.3";

TEST_CASE("Test Bayesian Classifiers score & version", "[BayesNet]")
{
    map <pair<std::string, std::string>, float> scores{
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
    std::map<std::string, bayesnet::BaseClassifier*> models{
        {"AODE", new bayesnet::AODE()}, {"AODELd", new bayesnet::AODELd()},
        {"BoostAODE", new bayesnet::BoostAODE()},
        {"KDB", new bayesnet::KDB(2)}, {"KDBLd", new bayesnet::KDBLd(2)},
        {"SPODE", new bayesnet::SPODE(1)}, {"SPODELd", new bayesnet::SPODELd(1)},
        {"TAN", new bayesnet::TAN()}, {"TANLd", new bayesnet::TANLd()}
    };
    std::string name = GENERATE("AODE", "AODELd", "KDB", "KDBLd", "SPODE", "SPODELd", "TAN", "TANLd");
    auto clf = models[name];

    SECTION("Test " + name + " classifier")
    {
        for (const std::string& file_name : { "glass", "iris", "ecoli", "diabetes" }) {
            auto clf = models[name];
            auto discretize = name.substr(name.length() - 2) != "Ld";
            auto raw = RawDatasets(file_name, discretize);
            clf->fit(raw.Xt, raw.yt, raw.featurest, raw.classNamet, raw.statest);
            auto score = clf->score(raw.Xt, raw.yt);
            INFO("File: " + file_name);
            REQUIRE(score == Catch::Approx(scores[{file_name, name}]).epsilon(raw.epsilon));
        }
    }
    SECTION("Library check version")
    {
        INFO("Checking version of " + name + " classifier");
        REQUIRE(clf->getVersion() == ACTUAL_VERSION);
    }
    delete clf;
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
    REQUIRE(clf.getNumberOfStates() == 19);
    REQUIRE(clf.getClassNumStates() == 3);
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
TEST_CASE("BoostAODE test used features in train note and score", "[BayesNet]")
{
    auto raw = RawDatasets("diabetes", true);
    auto clf = bayesnet::BoostAODE(true);
    clf.setHyperparameters({
        {"order", "asc"},
        {"convergence", true},
        {"repeatSparent",true},
        {"select_features","CFS"},
        });
    clf.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
    REQUIRE(clf.getNumberOfNodes() == 72);
    REQUIRE(clf.getNumberOfEdges() == 120);
    REQUIRE(clf.getNotes().size() == 3);
    REQUIRE(clf.getNotes()[0] == "Used features in initialization: 6 of 8 with CFS");
    REQUIRE(clf.getNotes()[1] == "Used features in train: 7 of 8");
    REQUIRE(clf.getNotes()[2] == "Number of models: 8");
    auto score = clf.score(raw.Xv, raw.yv);
    auto scoret = clf.score(raw.Xt, raw.yt);
    REQUIRE(score == Catch::Approx(0.8138).epsilon(raw.epsilon));
    REQUIRE(scoret == Catch::Approx(0.8138).epsilon(raw.epsilon));
}
TEST_CASE("Model predict_proba", "[BayesNet]")
{
    std::string model = GENERATE("TAN", "SPODE", "BoostAODEproba", "BoostAODEvoting");
    auto res_prob_tan = std::vector<std::vector<double>>({
    { 0.00375671, 0.994457, 0.00178621 },
    { 0.00137462, 0.992734, 0.00589123 },
    { 0.00137462, 0.992734, 0.00589123 },
    { 0.00137462, 0.992734, 0.00589123 },
    { 0.00218225, 0.992877, 0.00494094 },
    { 0.00494209, 0.0978534, 0.897205 },
    { 0.0054192, 0.974275, 0.0203054 },
    { 0.00433012, 0.985054, 0.0106159 },
    { 0.000860806, 0.996922, 0.00221698 }
        });
    auto res_prob_spode = std::vector<std::vector<double>>({
     {0.00419032, 0.994247, 0.00156265},
     {0.00172808, 0.993433, 0.00483862},
     {0.00172808, 0.993433, 0.00483862},
     {0.00172808, 0.993433, 0.00483862},
     {0.00279211, 0.993737, 0.00347077},
     {0.0120674, 0.357909, 0.630024},
     {0.00386239, 0.913919, 0.0822185},
     {0.0244389, 0.966447, 0.00911374},
     {0.003135, 0.991799, 0.0050661}
        });
    auto res_prob_baode = std::vector<std::vector<double>>({
        {0.00803291, 0.9676, 0.0243672},
        {0.00398714, 0.945126, 0.050887},
        {0.00398714, 0.945126, 0.050887},
        {0.00398714, 0.945126, 0.050887},
        {0.00189227, 0.859575, 0.138533},
        {0.0118341, 0.442149, 0.546017},
        {0.0216135, 0.785781, 0.192605},
        {0.0204803, 0.844276, 0.135244},
        {0.00576313, 0.961665, 0.0325716},
        });
    auto res_prob_voting = std::vector<std::vector<double>>({
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 0.447909, 0.552091},
        {0, 0.811482, 0.188517},
        {0, 1, 0},
        {0, 1, 0}
        });
    std::map<std::string, std::vector<std::vector<double>>> res_prob{ {"TAN", res_prob_tan}, {"SPODE", res_prob_spode} , {"BoostAODEproba", res_prob_baode }, {"BoostAODEvoting", res_prob_voting } };
    std::map<std::string, bayesnet::BaseClassifier*> models{ {"TAN", new bayesnet::TAN()}, {"SPODE", new bayesnet::SPODE(0)}, {"BoostAODEproba", new bayesnet::BoostAODE(false)}, {"BoostAODEvoting", new bayesnet::BoostAODE(true)} };
    int init_index = 78;
    auto raw = RawDatasets("iris", true);

    SECTION("Test " + model + " predict_proba")
    {
        auto clf = models[model];
        clf->fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
        auto y_pred_proba = clf->predict_proba(raw.Xv);
        auto yt_pred_proba = clf->predict_proba(raw.Xt);
        auto y_pred = clf->predict(raw.Xv);
        auto yt_pred = clf->predict(raw.Xt);
        REQUIRE(y_pred.size() == yt_pred.size(0));
        REQUIRE(y_pred.size() == y_pred_proba.size());
        REQUIRE(y_pred.size() == yt_pred_proba.size(0));
        REQUIRE(y_pred.size() == raw.yv.size());
        REQUIRE(y_pred_proba[0].size() == 3);
        REQUIRE(yt_pred_proba.size(1) == y_pred_proba[0].size());
        for (int i = 0; i < y_pred_proba.size(); ++i) {
            auto maxElem = max_element(y_pred_proba[i].begin(), y_pred_proba[i].end());
            int predictedClass = distance(y_pred_proba[i].begin(), maxElem);
            REQUIRE(predictedClass == y_pred[i]);
            // Check predict is coherent with predict_proba
            REQUIRE(yt_pred_proba[i].argmax().item<int>() == y_pred[i]);
            for (int j = 0; j < yt_pred_proba.size(1); j++) {
                REQUIRE(yt_pred_proba[i][j].item<double>() == Catch::Approx(y_pred_proba[i][j]).epsilon(raw.epsilon));
            }
        }
        // Check predict_proba values for vectors and tensors
        for (int i = 0; i < res_prob.size(); i++) {
            REQUIRE(y_pred[i] == yt_pred[i].item<int>());
            for (int j = 0; j < 3; j++) {
                REQUIRE(res_prob[model][i][j] == Catch::Approx(y_pred_proba[i + init_index][j]).epsilon(raw.epsilon));
                REQUIRE(res_prob[model][i][j] == Catch::Approx(yt_pred_proba[i + init_index][j].item<double>()).epsilon(raw.epsilon));
            }
        }
        delete clf;
    }
}
TEST_CASE("BoostAODE voting-proba", "[BayesNet]")
{
    auto raw = RawDatasets("iris", false);
    auto clf = bayesnet::BoostAODE(false);
    clf.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
    auto score_proba = clf.score(raw.Xv, raw.yv);
    auto pred_proba = clf.predict_proba(raw.Xv);
    clf.setHyperparameters({
        {"predict_voting",true},
        });
    auto score_voting = clf.score(raw.Xv, raw.yv);
    auto pred_voting = clf.predict_proba(raw.Xv);
    REQUIRE(score_proba == Catch::Approx(0.97333).epsilon(raw.epsilon));
    REQUIRE(score_voting == Catch::Approx(0.98).epsilon(raw.epsilon));
    REQUIRE(pred_voting[83][2] == Catch::Approx(0.552091).epsilon(raw.epsilon));
    REQUIRE(pred_proba[83][2] == Catch::Approx(0.546017).epsilon(raw.epsilon));
    clf.dump_cpt();
    REQUIRE(clf.topological_order() == std::vector<std::string>());
}
TEST_CASE("BoostAODE order asc, desc & random", "[BayesNet]")
{

    auto raw = RawDatasets("glass", true);
    std::map<std::string, double> scores{
        {"asc", 0.83178f }, { "desc", 0.84579f }, { "rand", 0.83645f }
    };
    for (const std::string& order : { "asc", "desc", "rand" }) {
        auto clf = bayesnet::BoostAODE();
        clf.setHyperparameters({
            {"order", order},
            });
        clf.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
        auto score = clf.score(raw.Xv, raw.yv);
        auto scoret = clf.score(raw.Xt, raw.yt);
        INFO("order: " + order);
        REQUIRE(score == Catch::Approx(scores[order]).epsilon(raw.epsilon));
        REQUIRE(scoret == Catch::Approx(scores[order]).epsilon(raw.epsilon));
    }
}
TEST_CASE("BoostAODE predict_single", "[BayesNet]")
{

    auto raw = RawDatasets("glass", true);
    std::map<bool, double> scores{
        {true, 0.84579f }, { false, 0.80841f }
    };
    for (const bool kind : { true, false}) {
        auto clf = bayesnet::BoostAODE();
        clf.setHyperparameters({
            {"predict_single", kind}, {"order", "desc" },
            });
        clf.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
        auto score = clf.score(raw.Xv, raw.yv);
        auto scoret = clf.score(raw.Xt, raw.yt);
        INFO("kind: " + std::string(kind ? "true" : "false"));
        REQUIRE(score == Catch::Approx(scores[kind]).epsilon(raw.epsilon));
        REQUIRE(scoret == Catch::Approx(scores[kind]).epsilon(raw.epsilon));
    }
}
