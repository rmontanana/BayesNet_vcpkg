#include <iostream>
#include <string>
#include <torch/torch.h>
#include <thread>
#include <argparse/argparse.hpp>
#include "ArffFiles.h"
#include "Network.h"
#include "BayesMetrics.h"
#include "CPPFImdlp.h"
#include "KDB.h"
#include "SPODE.h"
#include "AODE.h"
#include "TAN.h"
#include "platformUtils.h"
#include "Experiment.h"
#include "Folding.h"


using namespace std;

int main(int argc, char** argv)
{
    map<string, bool> datasets = {
            {"diabetes",           true},
            {"ecoli",              true},
            {"glass",              true},
            {"iris",               true},
            {"kdd_JapaneseVowels", false},
            {"letter",             true},
            {"liver-disorders",    true},
            {"mfeat-factors",      true},
    };
    auto valid_datasets = vector<string>();
    for (auto dataset : datasets) {
        valid_datasets.push_back(dataset.first);
    }
    argparse::ArgumentParser program("BayesNetSample");
    program.add_argument("-d", "--dataset")
        .help("Dataset file name")
        .action([valid_datasets](const std::string& value) {
        if (find(valid_datasets.begin(), valid_datasets.end(), value) != valid_datasets.end()) {
            return value;
        }
        throw runtime_error("file must be one of {diabetes, ecoli, glass, iris, kdd_JapaneseVowels, letter, liver-disorders, mfeat-factors}");
            }
    );
    program.add_argument("-p", "--path")
        .help("folder where the data files are located, default")
        .default_value(string{ PATH }
    );
    program.add_argument("-m", "--model")
        .help("Model to use {AODE, KDB, SPODE, TAN}")
        .action([](const std::string& value) {
        static const vector<string> choices = { "AODE", "KDB", "SPODE", "TAN" };
        if (find(choices.begin(), choices.end(), value) != choices.end()) {
            return value;
        }
        throw runtime_error("Model must be one of {AODE, KDB, SPODE, TAN}");
            }
    );
    program.add_argument("--title").required().help("Experiment title");
    program.add_argument("--discretize").help("Discretize input dataset").default_value(false).implicit_value(true);
    program.add_argument("--stratified").help("If Stratified KFold is to be done").default_value(false).implicit_value(true);
    program.add_argument("-f", "--folds").help("Number of folds").default_value(5).scan<'i', int>().action([](const string& value) {
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
    program.add_argument("-s", "--seed").help("Random seed").default_value(-1).scan<'i', int>();
    bool class_last, discretize_dataset, stratified;
    int n_folds, seed;
    string model_name, file_name, path, complete_file_name, title;
    try {
        program.parse_args(argc, argv);
        file_name = program.get<string>("dataset");
        path = program.get<string>("path");
        model_name = program.get<string>("model");
        discretize_dataset = program.get<bool>("discretize");
        stratified = program.get<bool>("stratified");
        n_folds = program.get<int>("folds");
        seed = program.get<int>("seed");
        complete_file_name = path + file_name + ".arff";
        class_last = datasets[file_name];
        title = program.get<string>("title");
        if (!file_exists(complete_file_name)) {
            throw runtime_error("Data File " + path + file_name + ".arff" + " does not exist");
        }
    }
    catch (const exception& err) {
        cerr << err.what() << endl;
        cerr << program;
        exit(1);
    }
    /*
    * Begin Processing
    */
    auto [X, y, features, className, states] = loadDataset(path, file_name, class_last, discretize_dataset);
    Fold* fold;
    if (stratified)
        fold = new StratifiedKFold(n_folds, y, seed);
    else
        fold = new KFold(n_folds, y.numel(), seed);
    auto experiment = platform::Experiment();
    experiment.setTitle(title).setLanguage("cpp").setLanguageVersion("1.0.0");
    experiment.setDiscretized(discretize_dataset).setModel(model_name).setModelVersion("1...0").setPlatform("BayesNet");
    experiment.setStratified(stratified).setNFolds(n_folds).addRandomSeed(seed).setScoreName("accuracy");
    platform::Timer timer;
    timer.start();
    auto result = platform::cross_validation(fold, model_name, X, y, features, className, states);
    result.setDataset(file_name);
    experiment.addResult(result);
    experiment.setDuration(timer.getDuration());
    experiment.save(path);
    experiment.show();
    return 0;
}
