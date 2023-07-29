#include <iostream>
#include <argparse/argparse.hpp>
#include "platformUtils.h"
#include "Experiment.h"
#include "Datasets.h"
#include "DotEnv.h"
#include "Models.h"


using namespace std;
const string PATH_RESULTS = "results";
const string PATH_DATASETS = "datasets";

argparse::ArgumentParser manageArguments(int argc, char** argv)
{
    auto env = platform::DotEnv();
    argparse::ArgumentParser program("BayesNetSample");
    program.add_argument("-d", "--dataset").default_value("").help("Dataset file name");
    program.add_argument("-p", "--path")
        .help("folder where the data files are located, default")
        .default_value(string{ PATH_DATASETS }
    );
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
    bool class_last, discretize_dataset, stratified;
    int n_folds;
    vector<int> seeds;
    string model_name, file_name, path, complete_file_name, title;
    try {
        program.parse_args(argc, argv);
        file_name = program.get<string>("dataset");
        path = program.get<string>("path");
        model_name = program.get<string>("model");
        discretize_dataset = program.get<bool>("discretize");
        stratified = program.get<bool>("stratified");
        n_folds = program.get<int>("folds");
        seeds = program.get<vector<int>>("seeds");
        complete_file_name = path + file_name + ".arff";
        class_last = false;//datasets[file_name];
        title = program.get<string>("title");
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
void registerModels()
{
    static platform::Registrar registrarT("TAN",
        [](void) -> bayesnet::BaseClassifier* { return new bayesnet::TAN();});
    static platform::Registrar registrarS("SPODE",
        [](void) -> bayesnet::BaseClassifier* { return new bayesnet::SPODE(2);});
    static platform::Registrar registrarK("KDB",
        [](void) -> bayesnet::BaseClassifier* { return new bayesnet::KDB(2);});
    static platform::Registrar registrarA("AODE",
        [](void) -> bayesnet::BaseClassifier* { return new bayesnet::AODE();});
}

int main(int argc, char** argv)
{
    registerModels();
    auto program = manageArguments(argc, argv);
    bool saveResults = false;
    auto file_name = program.get<string>("dataset");
    auto path = program.get<string>("path");
    auto model_name = program.get<string>("model");
    auto discretize_dataset = program.get<bool>("discretize");
    auto stratified = program.get<bool>("stratified");
    auto n_folds = program.get<int>("folds");
    auto seeds = program.get<vector<int>>("seeds");
    vector<string> filesToProcess;
    auto datasets = platform::Datasets(path, true, platform::ARFF);
    auto title = program.get<string>("title");
    if (file_name != "") {
        if (!datasets.isDataset(file_name)) {
            cerr << "Dataset " << file_name << " not found" << endl;
            exit(1);
        }
        if (title == "") {
            title = "Test " + file_name + " " + model_name + " " + to_string(n_folds) + " folds";
        }
        filesToProcess.push_back(file_name);
    } else {
        filesToProcess = platform::Datasets(path, true, platform::ARFF).getNames();
        saveResults = true;
    }

    /*
    * Begin Processing
    */
    auto experiment = platform::Experiment();
    experiment.setTitle(title).setLanguage("cpp").setLanguageVersion("1.0.0");
    experiment.setDiscretized(discretize_dataset).setModel(model_name).setPlatform("BayesNet");
    experiment.setStratified(stratified).setNFolds(n_folds).setScoreName("accuracy");
    for (auto seed : seeds) {
        experiment.addRandomSeed(seed);
    }
    platform::Timer timer;
    cout << "*** Starting experiment: " << title << " ***" << endl;
    timer.start();
    for (auto fileName : filesToProcess) {
        cout << "- " << setw(20) << left << fileName << " " << right << flush;
        auto result = experiment.cross_validation(path, fileName);
        result.setDataset(fileName);
        experiment.addResult(result);
        cout << endl;
    }
    experiment.setDuration(timer.getDuration());
    if (saveResults)
        experiment.save(PATH_RESULTS);
    else
        experiment.show();
    cout << "Done!" << endl;
    return 0;
}
