#include <iostream>
#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>
#include "platformUtils.h"
#include "Experiment.h"
#include "Datasets.h"
#include "DotEnv.h"
#include "Models.h"
#include "modelRegister.h"
#include "Paths.h"


using namespace std;
using json = nlohmann::json;

argparse::ArgumentParser manageArguments(int argc, char** argv)
{
    auto env = platform::DotEnv();
    argparse::ArgumentParser program("main");
    program.add_argument("-d", "--dataset").default_value("").help("Dataset file name");
    program.add_argument("--hyperparameters").default_value("{}").help("Hyperparamters passed to the model in Experiment");
    program.add_argument("-p", "--path")
        .help("folder where the data files are located, default")
        .default_value(string{ platform::Paths::datasets() });
    program.add_argument("-m", "--model")
        .help("Model to use " + platform::Models::instance()->toString())
        .action([](const std::string& value) {
        static const vector<string> choices = platform::Models::instance()->getNames();
        if (find(choices.begin(), choices.end(), value) != choices.end()) {
            return value;
        }
        throw runtime_error("Model must be one of " + platform::Models::instance()->toString());
            }
    );
    program.add_argument("--title").default_value("").help("Experiment title");
    program.add_argument("--discretize").help("Discretize input dataset").default_value((bool)stoi(env.get("discretize"))).implicit_value(true);
    program.add_argument("--save").help("Save result (always save if no dataset is supplied)").default_value(false).implicit_value(true);
    program.add_argument("--stratified").help("If Stratified KFold is to be done").default_value((bool)stoi(env.get("stratified"))).implicit_value(true);
    program.add_argument("-f", "--folds").help("Number of folds").default_value(stoi(env.get("n_folds"))).scan<'i', int>().action([](const string& value) {
        try {
            auto k = stoi(value);
            if (k < 2) {
                throw runtime_error("Number of folds must be greater than 1");
            }
            return k;
        }
        catch (const runtime_error& err) {
            throw runtime_error(err.what());
        }
        catch (...) {
            throw runtime_error("Number of folds must be an integer");
        }});
    auto seed_values = env.getSeeds();
    program.add_argument("-s", "--seeds").nargs(1, 10).help("Random seeds. Set to -1 to have pseudo random").scan<'i', int>().default_value(seed_values);
    try {
        program.parse_args(argc, argv);
        auto file_name = program.get<string>("dataset");
        auto path = program.get<string>("path");
        auto model_name = program.get<string>("model");
        auto discretize_dataset = program.get<bool>("discretize");
        auto stratified = program.get<bool>("stratified");
        auto n_folds = program.get<int>("folds");
        auto seeds = program.get<vector<int>>("seeds");
        auto complete_file_name = path + file_name + ".arff";
        auto title = program.get<string>("title");
        auto hyperparameters = program.get<string>("hyperparameters");
        auto saveResults = program.get<bool>("save");
        if (title == "" && file_name == "") {
            throw runtime_error("title is mandatory if dataset is not provided");
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
    auto file_name = program.get<string>("dataset");
    auto path = program.get<string>("path");
    auto model_name = program.get<string>("model");
    auto discretize_dataset = program.get<bool>("discretize");
    auto stratified = program.get<bool>("stratified");
    auto n_folds = program.get<int>("folds");
    auto seeds = program.get<vector<int>>("seeds");
    auto hyperparameters = program.get<string>("hyperparameters");
    vector<string> filesToTest;
    auto env = platform::DotEnv();
    auto datasets = platform::Datasets(discretize_dataset, env.get("source_data"));
    auto title = program.get<string>("title");
    auto saveResults = program.get<bool>("save");
    if (file_name != "") {
        if (!datasets.isDataset(file_name)) {
            cerr << "Dataset " << file_name << " not found" << endl;
            exit(1);
        }
        if (title == "") {
            title = "Test " + file_name + " " + model_name + " " + to_string(n_folds) + " folds";
        }
        filesToTest.push_back(file_name);
    } else {
        filesToTest = datasets.getNames();
        saveResults = true;
    }
    /*
    * Begin Processing
    */

    auto experiment = platform::Experiment();
    experiment.setTitle(title).setLanguage("cpp").setLanguageVersion("14.0.3");
    experiment.setDiscretized(discretize_dataset).setModel(model_name).setPlatform(env.get("platform"));
    experiment.setStratified(stratified).setNFolds(n_folds).setScoreName("accuracy");
    experiment.setHyperparameters(json::parse(hyperparameters));
    for (auto seed : seeds) {
        experiment.addRandomSeed(seed);
    }
    platform::Timer timer;
    timer.start();
    experiment.go(filesToTest, path);
    experiment.setDuration(timer.getDuration());
    if (saveResults) {
        experiment.save(platform::Paths::results());
    }
    experiment.report();
    cout << "Done!" << endl;
    return 0;
}
