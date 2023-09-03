#include <iostream>
#include <argparse/argparse.hpp>
#include "platformUtils.h"
#include "Paths.h"
#include "Results.h"

using namespace std;

argparse::ArgumentParser manageArguments(int argc, char** argv)
{
    argparse::ArgumentParser program("manage");
    program.add_argument("-n", "--number").default_value(0).help("Number of results to show (0 = all)").scan<'i', int>();
    program.add_argument("-m", "--model").default_value("any").help("Filter results of the selected model)");
    program.add_argument("-s", "--score").default_value("any").help("Filter results of the score name supplied");
    program.add_argument("--complete").help("Show only results with all datasets").default_value(false).implicit_value(true);
    try {
        program.parse_args(argc, argv);
        auto number = program.get<int>("number");
        if (number < 0) {
            throw runtime_error("Number of results must be greater than or equal to 0");
        }
        auto model = program.get<string>("model");
        auto score = program.get<string>("score");
        auto complete = program.get<bool>("complete");
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
    auto number = program.get<int>("number");
    auto model = program.get<string>("model");
    auto score = program.get<string>("score");
    auto complete = program.get<bool>("complete");
    auto results = platform::Results(platform::Paths::results(), number, model, score, complete);
    results.manage();
    return 0;
}
