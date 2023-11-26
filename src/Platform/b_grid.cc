#include <iostream>
#include <argparse/argparse.hpp>
#include "DotEnv.h"
#include "Models.h"
#include "modelRegister.h"
#include "GridSearch.h"
#include "Paths.h"
#include "Timer.h"


argparse::ArgumentParser manageArguments(std::string program_name)
{
    auto env = platform::DotEnv();
    argparse::ArgumentParser program(program_name);
    program.add_argument("-m", "--model")
        .help("Model to use " + platform::Models::instance()->tostring())
        .action([](const std::string& value) {
        static const std::vector<std::string> choices = platform::Models::instance()->getNames();
        if (find(choices.begin(), choices.end(), value) != choices.end()) {
            return value;
        }
        throw std::runtime_error("Model must be one of " + platform::Models::instance()->tostring());
            }
    );
    program.add_argument("--discretize").help("Discretize input datasets").default_value((bool)stoi(env.get("discretize"))).implicit_value(true);
    program.add_argument("--quiet").help("Don't display detailed progress").default_value(false).implicit_value(true);
    program.add_argument("--continue").help("Continue computing from that dataset").default_value("No");
    program.add_argument("--stratified").help("If Stratified KFold is to be done").default_value((bool)stoi(env.get("stratified"))).implicit_value(true);
    program.add_argument("--score").help("Score used in gridsearch").default_value("accuracy");
    program.add_argument("-f", "--folds").help("Number of folds").default_value(stoi(env.get("n_folds"))).scan<'i', int>().action([](const std::string& value) {
        try {
            auto k = stoi(value);
            if (k < 2) {
                throw std::runtime_error("Number of folds must be greater than 1");
            }
            return k;
        }
        catch (const runtime_error& err) {
            throw std::runtime_error(err.what());
        }
        catch (...) {
            throw std::runtime_error("Number of folds must be an integer");
        }});
    auto seed_values = env.getSeeds();
    program.add_argument("-s", "--seeds").nargs(1, 10).help("Random seeds. Set to -1 to have pseudo random").scan<'i', int>().default_value(seed_values);
    return program;
}

int main(int argc, char** argv)
{
    auto program = manageArguments("b_grid");
    struct platform::ConfigGrid config;
    try {
        program.parse_args(argc, argv);
        config.model = program.get<std::string>("model");
        config.score = program.get<std::string>("score");
        config.discretize = program.get<bool>("discretize");
        config.stratified = program.get<bool>("stratified");
        config.n_folds = program.get<int>("folds");
        config.quiet = program.get<bool>("quiet");
        config.seeds = program.get<std::vector<int>>("seeds");
        config.continue_from = program.get<std::string>("continue");
    }
    catch (const exception& err) {
        cerr << err.what() << std::endl;
        cerr << program;
        exit(1);
    }
    /*
     * Begin Processing
     */
    auto env = platform::DotEnv();
    platform::Paths::createPath(platform::Paths::grid());
    config.path = platform::Paths::grid();
    auto grid_search = platform::GridSearch(config);
    platform::Timer timer;
    timer.start();
    grid_search.go();
    std::cout << "Process took " << timer.getDurationString() << std::endl;
    std::cout << "Done!" << std::endl;
    return 0;
}
