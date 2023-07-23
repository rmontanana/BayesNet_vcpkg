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
#include "Folding.h"


using namespace std;

pair<float, float> cross_validation(Fold* fold, bayesnet::BaseClassifier* model, Tensor& X, Tensor& y, vector<string> features, string className, map<string, vector<int>> states)
{
    auto k = fold->getNumberOfFolds();
    float accuracy = 0.0;
    for (int i = 0; i < k; i++) {
        auto [train, test] = fold->getFold(i);
        auto train_t = torch::tensor(train);
        auto test_t = torch::tensor(test);
        auto X_train = X.index({ train_t });
        auto y_train = y.index({ train_t });
        auto X_test = X.index({ test_t });
        auto y_test = y.index({ test_t });
        model->fit(X_train, y_train, features, className, states);
        auto acc = model->score(X_test, y_test);
        accuracy += acc;
    }
    return { accuracy / k, 0 };
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
    program.add_argument("-f", "--file")
        .help("Dataset file name")
        .action([valid_datasets](const std::string& value) {
        if (find(valid_datasets.begin(), valid_datasets.end(), value) != valid_datasets.end()) {
            return value;
        }
        throw runtime_error("file must be one of {diabetes, ecoli, glass, iris, kdd_JapaneseVowels, letter, liver-disorders, mfeat-factors}");
            }
    );
    program.add_argument("-p", "--path")
        .help(" folder where the data files are located, default")
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
    program.add_argument("--discretize").default_value(false).implicit_value(true);
    bool class_last, discretize_dataset;
    string model_name, file_name, path, complete_file_name;
    try {
        program.parse_args(argc, argv);
        file_name = program.get<string>("file");
        path = program.get<string>("path");
        model_name = program.get<string>("model");
        discretize_dataset = program.get<bool>("discretize");
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
    auto [X, y, features, className] = loadDataset(file_name, discretize_dataset, class_last);
    auto states = map<string, vector<int>>();
    if (discretize_dataset) {
        auto [Xd, maxes] = discretizeTorch(X, y, features);
        states = get_states(Xd, y, features, className);
        X = Xd;
    }
    auto fold = StratifiedKFold(5, y, -1);
    auto classifiers = map<string, bayesnet::BaseClassifier*>({
        { "AODE", new bayesnet::AODE() }, { "KDB", new bayesnet::KDB(2) },
        { "SPODE",  new bayesnet::SPODE(2) }, { "TAN",  new bayesnet::TAN() }
        }
    );
    bayesnet::BaseClassifier* model = classifiers[model_name];
    auto results = cross_validation(&fold, model, X, y, features, className, states);
    cout << "Accuracy: " << results.first << endl;
    return 0;
}
