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


using namespace std;

const string PATH = "../../data/";

inline constexpr auto hash_conv(const std::string_view sv)
{
    unsigned long hash{ 5381 };
    for (unsigned char c : sv) {
        hash = ((hash << 5) + hash) ^ c;
    }
    return hash;
}

inline constexpr auto operator"" _sh(const char* str, size_t len)
{
    return hash_conv(std::string_view{ str, len });
}

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
    auto handler = ArffFiles();
    handler.load(complete_file_name, class_last);
    // Get Dataset X, y
    vector<mdlp::samples_t>& X = handler.getX();
    mdlp::labels_t& y = handler.getY();
    // Get className & Features
    auto className = handler.getClassName();
    vector<string> features;
    for (auto feature : handler.getAttributes()) {
        features.push_back(feature.first);
    }
    // Discretize Dataset
    vector<mdlp::labels_t> Xd;
    map<string, int> maxes;
    tie(Xd, maxes) = discretize(X, y, features);
    maxes[className] = *max_element(y.begin(), y.end()) + 1;
    map<string, vector<int>> states;
    for (auto feature : features) {
        states[feature] = vector<int>(maxes[feature]);
    }
    states[className] = vector<int>(
        maxes[className]);
    double score;
    vector<string> lines;
    vector<string> graph;
    auto kdb = bayesnet::KDB(2);
    auto aode = bayesnet::AODE();
    auto spode = bayesnet::SPODE(2);
    auto tan = bayesnet::TAN();
    switch (hash_conv(model_name)) {
        case "AODE"_sh:
            aode.fit(Xd, y, features, className, states);
            lines = aode.show();
            score = aode.score(Xd, y);
            graph = aode.graph();
            break;
        case "KDB"_sh:
            kdb.fit(Xd, y, features, className, states);
            lines = kdb.show();
            score = kdb.score(Xd, y);
            graph = kdb.graph();
            break;
        case "SPODE"_sh:
            spode.fit(Xd, y, features, className, states);
            lines = spode.show();
            score = spode.score(Xd, y);
            graph = spode.graph();
            break;
        case "TAN"_sh:
            tan.fit(Xd, y, features, className, states);
            lines = tan.show();
            score = tan.score(Xd, y);
            graph = tan.graph();
            break;
    }
    for (auto line : lines) {
        cout << line << endl;
    }
    cout << "Score: " << score << endl;
    auto dot_file = model_name + "_" + file_name;
    ofstream file(dot_file + ".dot");
    file << graph;
    file.close();
    cout << "Graph saved in " << model_name << "_" << file_name << ".dot" << endl;
    cout << "dot -Tpng -o " + dot_file + ".png " + dot_file + ".dot " << endl;
    return 0;
}