#include <iostream>
#include <argparse/argparse.hpp>
#include "Paths.h"
#include "BestResults.h"
#include "Colors.h"

using namespace std;

argparse::ArgumentParser manageArguments(int argc, char** argv)
{
    argparse::ArgumentParser program("best");
    program.add_argument("-m", "--model").default_value("").help("Filter results of the selected model) (any for all models)");
    program.add_argument("-s", "--score").default_value("").help("Filter results of the score name supplied");
    program.add_argument("--build").help("build best score results file").default_value(false).implicit_value(true);
    program.add_argument("--report").help("report of best score results file").default_value(false).implicit_value(true);
    program.add_argument("--friedman").help("Friedman test").default_value(false).implicit_value(true);
    program.add_argument("--excel").help("Output to excel").default_value(false).implicit_value(true);
    try {
        program.parse_args(argc, argv);
        auto model = program.get<string>("model");
        auto score = program.get<string>("score");
        auto build = program.get<bool>("build");
        auto report = program.get<bool>("report");
        auto friedman = program.get<bool>("friedman");
        auto excel = program.get<bool>("excel");
        if (model == "" || score == "") {
            throw runtime_error("Model and score name must be supplied");
        }
        if (friedman && model != "any") {
            cerr << "Friedman test can only be used with all models" << endl;
            cerr << program;
            exit(1);
        }
        if (excel && model != "any") {
            cerr << "Excel ourput can only be used with all models" << endl;
            cerr << program;
            exit(1);
        }
        if (!report && !build) {
            cerr << "Either build, report or both, have to be selected to do anything!" << endl;
            cerr << program;
            exit(1);
        }
    }
    catch (const exception& err) {
        cerr << err.what() << endl;
        cerr << program;
        exit(1);
    }
    return program;
}

int main(int argc, char** argv)
{
    auto program = manageArguments(argc, argv);
    auto model = program.get<string>("model");
    auto score = program.get<string>("score");
    auto build = program.get<bool>("build");
    auto report = program.get<bool>("report");
    auto friedman = program.get<bool>("friedman");
    auto excel = program.get<bool>("excel");
    auto results = platform::BestResults(platform::Paths::results(), score, model, friedman);
    if (build) {
        if (model == "any") {
            results.buildAll();
        } else {
            string fileName = results.build();
            cout << Colors::GREEN() << fileName << " created!" << Colors::RESET() << endl;
        }
    }
    if (report) {
        if (model == "any") {
            results.reportAll(excel);
        } else {
            results.reportSingle();
        }
    }
    return 0;
}
