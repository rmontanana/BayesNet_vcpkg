#include <iostream>
#include <torch/torch.h>
#include <string>
#include <map>
#include <argparse/argparse.hpp>
#include "ArffFiles.h"
#include "BayesMetrics.h"
#include "CPPFImdlp.h"
#include "Folding.h"
#include "Models.h"
#include "modelRegister.h"


using namespace std;

const string PATH = "../../data/";

pair<vector<mdlp::labels_t>, map<string, int>> discretize(vector<mdlp::samples_t>& X, mdlp::labels_t& y, vector<string> features)
{
    vector<mdlp::labels_t>Xd;
    map<string, int> maxes;

    auto fimdlp = mdlp::CPPFImdlp();
    for (int i = 0; i < X.size(); i++) {
        fimdlp.fit(X[i], y);
        mdlp::labels_t& xd = fimdlp.transform(X[i]);
        maxes[features[i]] = *max_element(xd.begin(), xd.end()) + 1;
        Xd.push_back(xd);
    }
    return { Xd, maxes };
}

bool file_exists(const std::string& name)
{
    if (FILE* file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}
pair<vector<vector<int>>, vector<int>> extract_indices(vector<int> indices, vector<vector<int>> X, vector<int> y)
{
    vector<vector<int>> Xr; // nxm
    vector<int> yr;
    for (int col = 0; col < X.size(); ++col) {
        Xr.push_back(vector<int>());
    }
    for (auto index : indices) {
        for (int col = 0; col < X.size(); ++col) {
            Xr[col].push_back(X[col][index]);
        }
        yr.push_back(y[index]);
    }
    return { Xr, yr };
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
    transform(datasets.begin(), datasets.end(), back_inserter(valid_datasets),
        [](const pair<string, bool>& pair) { return pair.first; });
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
        .help(" folder where the data files are located, default")
        .default_value(string{ PATH }
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
    program.add_argument("--discretize").help("Discretize input dataset").default_value(false).implicit_value(true);
    program.add_argument("--dumpcpt").help("Dump CPT Tables").default_value(false).implicit_value(true);
    program.add_argument("--stratified").help("If Stratified KFold is to be done").default_value(false).implicit_value(true);
    program.add_argument("--tensors").help("Use tensors to store samples").default_value(false).implicit_value(true);
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
    bool class_last, stratified, tensors, dump_cpt;
    string model_name, file_name, path, complete_file_name;
    int nFolds, seed;
    try {
        program.parse_args(argc, argv);
        file_name = program.get<string>("dataset");
        path = program.get<string>("path");
        model_name = program.get<string>("model");
        complete_file_name = path + file_name + ".arff";
        stratified = program.get<bool>("stratified");
        tensors = program.get<bool>("tensors");
        nFolds = program.get<int>("folds");
        seed = program.get<int>("seed");
        dump_cpt = program.get<bool>("dumpcpt");
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
    auto handler = ArffFiles();
    handler.load(complete_file_name, class_last);
    // Get Dataset X, y
    vector<mdlp::samples_t>& X = handler.getX();
    mdlp::labels_t& y = handler.getY();
    // Get className & Features
    auto className = handler.getClassName();
    vector<string> features;
    auto attributes = handler.getAttributes();
    transform(attributes.begin(), attributes.end(), back_inserter(features),
        [](const pair<string, string>& item) { return item.first; });
    // Discretize Dataset
    auto [Xd, maxes] = discretize(X, y, features);
    maxes[className] = *max_element(y.begin(), y.end()) + 1;
    map<string, vector<int>> states;
    for (auto feature : features) {
        states[feature] = vector<int>(maxes[feature]);
    }
    states[className] = vector<int>(maxes[className]);
    auto clf = platform::Models::instance()->create(model_name);
    clf->fit(Xd, y, features, className, states);
    if (dump_cpt) {
        cout << "--- CPT Tables ---" << endl;
        clf->dump_cpt();
    }
    auto lines = clf->show();
    for (auto line : lines) {
        cout << line << endl;
    }
    cout << "--- Topological Order ---" << endl;
    auto order = clf->topological_order();
    for (auto name : order) {
        cout << name << ", ";
    }
    cout << "end." << endl;
    auto score = clf->score(Xd, y);
    cout << "Score: " << score << endl;
    auto graph = clf->graph();
    auto dot_file = model_name + "_" + file_name;
    ofstream file(dot_file + ".dot");
    file << graph;
    file.close();
    cout << "Graph saved in " << model_name << "_" << file_name << ".dot" << endl;
    cout << "dot -Tpng -o " + dot_file + ".png " + dot_file + ".dot " << endl;
    string stratified_string = stratified ? " Stratified" : "";
    cout << nFolds << " Folds" << stratified_string << " Cross validation" << endl;
    cout << "==========================================" << endl;
    torch::Tensor Xt = torch::zeros({ static_cast<int>(Xd.size()), static_cast<int>(Xd[0].size()) }, torch::kInt32);
    torch::Tensor yt = torch::tensor(y, torch::kInt32);
    for (int i = 0; i < features.size(); ++i) {
        Xt.index_put_({ i, "..." }, torch::tensor(Xd[i], torch::kInt32));
    }
    float total_score = 0, total_score_train = 0, score_train, score_test;
    Fold* fold;
    if (stratified)
        fold = new StratifiedKFold(nFolds, y, seed);
    else
        fold = new KFold(nFolds, y.size(), seed);
    for (auto i = 0; i < nFolds; ++i) {
        auto [train, test] = fold->getFold(i);
        cout << "Fold: " << i + 1 << endl;
        if (tensors) {
            auto ttrain = torch::tensor(train, torch::kInt64);
            auto ttest = torch::tensor(test, torch::kInt64);
            torch::Tensor Xtraint = torch::index_select(Xt, 1, ttrain);
            torch::Tensor ytraint = yt.index({ ttrain });
            torch::Tensor Xtestt = torch::index_select(Xt, 1, ttest);
            torch::Tensor ytestt = yt.index({ ttest });
            clf->fit(Xtraint, ytraint, features, className, states);
            auto temp = clf->predict(Xtraint);
            score_train = clf->score(Xtraint, ytraint);
            score_test = clf->score(Xtestt, ytestt);
        } else {
            auto [Xtrain, ytrain] = extract_indices(train, Xd, y);
            auto [Xtest, ytest] = extract_indices(test, Xd, y);
            clf->fit(Xtrain, ytrain, features, className, states);

            score_train = clf->score(Xtrain, ytrain);
            score_test = clf->score(Xtest, ytest);
        }
        if (dump_cpt) {
            cout << "--- CPT Tables ---" << endl;
            clf->dump_cpt();
        }
        total_score_train += score_train;
        total_score += score_test;
        cout << "Score Train: " << score_train << endl;
        cout << "Score Test : " << score_test << endl;
        cout << "-------------------------------------------------------------------------------" << endl;
    }
    cout << "**********************************************************************************" << endl;
    cout << "Average Score Train: " << total_score_train / nFolds << endl;
    cout << "Average Score Test : " << total_score / nFolds << endl;
    return 0;
}