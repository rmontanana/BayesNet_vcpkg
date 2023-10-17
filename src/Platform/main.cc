#include <iostream>
#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>
#include "Experiment.h"
#include "Datasets.h"
#include "DotEnv.h"
#include "Models.h"
#include "modelRegister.h"
#include "Paths.h"


using namespace std;
using json = nlohmann::json;

argparse::ArgumentParser manageArguments()
{
    auto env = platform::DotEnv();
    argparse::ArgumentParser program("main");
    program.add_argument("-d", "--dataset").default_value("").help("Dataset file name");
    program.add_argument("--hyperparameters").default_value("{}").help("Hyperparamters passed to the model in Experiment");
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
    program.add_argument("--quiet").help("Don't display detailed progress").default_value(false).implicit_value(true);
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
    return program;
}

int main(int argc, char** argv)
{
    string file_name, model_name, title;
    json hyperparameters_json;
    bool discretize_dataset, stratified, saveResults, quiet;
    vector<int> seeds;
    vector<string> filesToTest;
    int n_folds;
    auto program = manageArguments();
    try {
        program.parse_args(argc, argv);
        file_name = program.get<string>("dataset");
        model_name = program.get<string>("model");
        discretize_dataset = program.get<bool>("discretize");
        stratified = program.get<bool>("stratified");
        quiet = program.get<bool>("quiet");
        n_folds = program.get<int>("folds");
        seeds = program.get<vector<int>>("seeds");
        auto hyperparameters = program.get<string>("hyperparameters");
        hyperparameters_json = json::parse(hyperparameters);
        title = program.get<string>("title");
        if (title == "" && file_name == "") {
            throw runtime_error("title is mandatory if dataset is not provided");
        }
        saveResults = program.get<bool>("save");
    }
    catch (const exception& err) {
        cerr << err.what() << endl;
        cerr << program;
        exit(1);
    }
    auto datasets = platform::Datasets(discretize_dataset, platform::Paths::datasets());
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
    auto env = platform::DotEnv();
    auto experiment = platform::Experiment();
    experiment.setTitle(title).setLanguage("cpp").setLanguageVersion("14.0.3");
    experiment.setDiscretized(discretize_dataset).setModel(model_name).setPlatform(env.get("platform"));
    experiment.setStratified(stratified).setNFolds(n_folds).setScoreName("accuracy");
    experiment.setHyperparameters(hyperparameters_json);
    for (auto seed : seeds) {
        experiment.addRandomSeed(seed);
    }
    platform::Timer timer;
    timer.start();
    experiment.go(filesToTest, quiet);
    experiment.setDuration(timer.getDuration());
    if (saveResults) {
        experiment.save(platform::Paths::results());
    }
    if (!quiet)
        experiment.report();
    cout << "Done!" << endl;
    return 0;
}
