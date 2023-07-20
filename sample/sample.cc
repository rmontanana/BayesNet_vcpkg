#include <iostream>
#include <string>
#include <torch/torch.h>
#include <thread>
#include <getopt.h>
#include "ArffFiles.h"
#include "Network.h"
#include "BayesMetrics.h"
#include "CPPFImdlp.h"
#include "KDB.h"
#include "SPODE.h"
#include "AODE.h"
#include "TAN.h"


using namespace std;

const string PATH = "data/";

/* print a description of all supported options */
void usage(const char* path)
{
    /* take only the last portion of the path */
    const char* basename = strrchr(path, '/');
    basename = basename ? basename + 1 : path;

    cout << "usage: " << basename << "[OPTION]" << endl;
    cout << "  -h, --help\t\t Print this help and exit." << endl;
    cout
        << "  -f, --file[=FILENAME]\t {diabetes, glass, iris, kdd_JapaneseVowels, letter, liver-disorders, mfeat-factors}."
        << endl;
    cout << "  -p, --path[=FILENAME]\t folder where the data files are located, default " << PATH << endl;
    cout << "  -m, --model={AODE, KDB, SPODE, TAN}\t " << endl;
}

tuple<string, string, string> parse_arguments(int argc, char** argv)
{
    string file_name;
    string model_name;
    string path = PATH;
    const vector<struct option> long_options = {
            {"help",          no_argument,       nullptr, 'h'},
            {"file",          required_argument, nullptr, 'f'},
            {"path",          required_argument, nullptr, 'p'},
            {"model",         required_argument, nullptr, 'm'},
            {nullptr,         no_argument,       nullptr, 0}
    };
    while (true) {
        const auto c = getopt_long(argc, argv, "hf:p:m:", long_options.data(), nullptr);
        if (c == -1)
            break;
        switch (c) {
            case 'h':
                usage(argv[0]);
                exit(0);
            case 'f':
                file_name = string(optarg);
                break;
            case 'm':
                model_name = string(optarg);
                break;
            case 'p':
                path = optarg;
                if (path.back() != '/')
                    path += '/';
                break;
            case '?':
                usage(argv[0]);
                exit(1);
            default:
                abort();
        }
    }
    if (file_name.empty()) {
        usage(argv[0]);
        exit(1);
    }
    return make_tuple(file_name, path, model_name);
}

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

tuple<string, string, string> get_options(int argc, char** argv)
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
    vector <string> models = { "AODE", "KDB", "SPODE", "TAN" };
    string file_name;
    string path;
    string model_name;
    tie(file_name, path, model_name) = parse_arguments(argc, argv);
    if (datasets.find(file_name) == datasets.end()) {
        cout << "Invalid file name: " << file_name << endl;
        usage(argv[0]);
        exit(1);
    }
    if (!file_exists(path + file_name + ".arff")) {
        cout << "Data File " << path + file_name + ".arff" << " does not exist" << endl;
        usage(argv[0]);
        exit(1);
    }
    if (find(models.begin(), models.end(), model_name) == models.end()) {
        cout << "Invalid model name: " << model_name << endl;
        usage(argv[0]);
        exit(1);
    }
    return { file_name, path, model_name };
}

int main(int argc, char** argv)
{
    string file_name, path, model_name;
    tie(file_name, path, model_name) = get_options(argc, argv);
    auto handler = ArffFiles();
    handler.load(path + file_name + ".arff");
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