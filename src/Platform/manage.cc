#include <iostream>
#include <argparse/argparse.hpp>
#include "platformUtils.h"
#include "Results.h"

using namespace std;
const string PATH_RESULTS = "results";

argparse::ArgumentParser manageArguments(int argc, char** argv)
{
    argparse::ArgumentParser program("manage");
    program.add_argument("-n", "--number").default_value(0).help("Number of results to show (0 = all)").scan<'i', int>();
    try {
        program.parse_args(argc, argv);
        auto number = program.get<int>("number");
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
    auto results = platform::Results(PATH_RESULTS);
    results.manage();
    return 0;
}
