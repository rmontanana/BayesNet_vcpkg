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
#include "Result.h"
#include "Folding.h"


using namespace std;

Result cross_validation(Fold* fold, string model_name, Tensor& X, Tensor& y, vector<string> features, string className, map<string, vector<int>> states)
{
    auto classifiers = map<string, bayesnet::BaseClassifier*>({
       { "AODE", new bayesnet::AODE() }, { "KDB", new bayesnet::KDB(2) },
       { "SPODE",  new bayesnet::SPODE(2) }, { "TAN",  new bayesnet::TAN() }
        }
    );
    auto result = Result();
    auto k = fold->getNumberOfFolds();
    auto accuracy = torch::zeros({ k }, kFloat64);
    auto train_time = torch::zeros({ k }, kFloat64);
    auto test_time = torch::zeros({ k }, kFloat64);
    Timer train_timer, test_timer;
    for (int i = 0; i < k; i++) {
        bayesnet::BaseClassifier* model = classifiers[model_name];
        train_timer.start();
        auto [train, test] = fold->getFold(i);
        auto train_t = torch::tensor(train);
        auto test_t = torch::tensor(test);
        auto X_train = X.index({ train_t, "..." });
        auto y_train = y.index({ train_t });
        auto X_test = X.index({ test_t, "..." });
        auto y_test = y.index({ test_t });
        model->fit(X_train, y_train, features, className, states);
        cout << "Training Fold " << i + 1 << endl;
        cout << "X_train: " << X_train.sizes() << endl;
        cout << "y_train: " << y_train.sizes() << endl;
        cout << "X_test: " << X_test.sizes() << endl;
        cout << "y_test: " << y_test.sizes() << endl;
        train_time[i] = train_timer.getDuration();
        test_timer.start();
        auto acc = model->score(X_test, y_test);
        test_time[i] = test_timer.getDuration();
        accuracy[i] = acc;
    }
    result.setScore(torch::mean(accuracy).item<double>());
    result.setTrainTime(torch::mean(train_time).item<double>()).setTestTime(torch::mean(test_time).item<double>());
    return result;
}

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
    bool class_last, discretize_dataset, stratified;
    int n_folds;
    string model_name, file_name, path, complete_file_name;
    try {
        program.parse_args(argc, argv);
        file_name = program.get<string>("dataset");
        path = program.get<string>("path");
        model_name = program.get<string>("model");
        discretize_dataset = program.get<bool>("discretize");
        stratified = program.get<bool>("stratified");
        n_folds = program.get<int>("folds");
        complete_file_name = path + file_name + ".arff";
        class_last = datasets[file_name];
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
        fold = new StratifiedKFold(n_folds, y, -1);
    else
        fold = new KFold(n_folds, y.numel(), -1);

    auto experiment = Experiment();
    experiment.setDiscretized(discretize_dataset).setModel(model_name).setPlatform("cpp");
    experiment.setStratified(stratified).setNFolds(5).addRandomSeed(271).setScoreName("accuracy");
    auto result = cross_validation(fold, model_name, X, y, features, className, states);
    result.setDataset(file_name);
    experiment.addResult(result);
    experiment.save(path);
    for (auto& item : states) {
        cout << item.first << ": " << item.second.size() << endl;
    }
    return 0;
}
