#include <iostream>
#include <argparse/argparse.hpp>
#include "platformUtils.h"
#include "Experiment.h"
#include "Datasets.h"


using namespace std;
const string PATH_RESULTS = "results";

argparse::ArgumentParser manageArguments(int argc, char** argv)
{
    argparse::ArgumentParser program("BayesNetSample");
    program.add_argument("-d", "--dataset").default_value("").help("Dataset file name");
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
        class_last = false;//datasets[file_name];
        title = program.get<string>("title");
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
    auto seed = program.get<int>("seed");
    vector<string> filesToProcess;
    auto datasets = platform::Datasets(path, true, platform::ARFF);
    if (file_name != "") {
        if (!datasets.isDataset(file_name)) {
            cerr << "Dataset " << file_name << " not found" << endl;
            exit(1);
        }
        filesToProcess.push_back(file_name);
    } else {
        filesToProcess = platform::Datasets(path, true, platform::ARFF).getNames();
    }
    auto title = program.get<string>("title");

    /*
    * Begin Processing
    */
    auto experiment = platform::Experiment();
    experiment.setTitle(title).setLanguage("cpp").setLanguageVersion("1.0.0");
    experiment.setDiscretized(discretize_dataset).setModel(model_name).setPlatform("BayesNet");
    experiment.setStratified(stratified).setNFolds(n_folds).addRandomSeed(seed).setScoreName("accuracy");
    platform::Timer timer;
    cout << "*** Starting experiment: " << title << " ***" << endl;
    timer.start();
    for (auto fileName : filesToProcess) {
        cout << "- " << setw(20) << left << fileName << " " << right << flush;
        auto [X, y] = datasets.getTensors(fileName);
        auto states = datasets.getStates(fileName);
        auto features = datasets.getFeatures(fileName);
        auto samples = datasets.getNSamples(fileName);
        auto className = datasets.getClassName(fileName);
        cout << " (" << setw(5) << samples << "," << setw(3) << features.size() << ") " << flush;
        Fold* fold;
        if (stratified)
            fold = new StratifiedKFold(n_folds, y, seed);
        else
            fold = new KFold(n_folds, samples, seed);
        auto result = platform::cross_validation(fold, model_name, X, y, features, className, states);
        result.setDataset(fileName);
        experiment.setModelVersion(result.getModelVersion());
        experiment.addResult(result);
        delete fold;
    }
    experiment.setDuration(timer.getDuration());
    experiment.save(PATH_RESULTS);
    cout << "Done!" << endl;
    return 0;
}
