#include <iostream>
#include <string>
#include <torch/torch.h>
#include <thread>
#include <getopt.h>
#include "ArffFiles.h"
#include "Network.h"
#include "CPPFImdlp.h"


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
    cout << "  -n, --net=[FILENAME]\t  default=file parameter value" << endl;
}

tuple<string, string, string> parse_arguments(int argc, char** argv)
{
    string file_name;
    string network_name;
    string path = PATH;
    const vector<struct option> long_options = {
            {"help",          no_argument,       nullptr, 'h'},
            {"file",          required_argument, nullptr, 'f'},
            {"path",          required_argument, nullptr, 'p'},
            {"net",           required_argument, nullptr, 'n'},
            {nullptr,         no_argument,       nullptr, 0}
    };
    while (true) {
        const auto c = getopt_long(argc, argv, "hf:p:n:", long_options.data(), nullptr);
        if (c == -1)
            break;
        switch (c) {
            case 'h':
                usage(argv[0]);
                exit(0);
            case 'f':
                file_name = string(optarg);
                break;
            case 'n':
                network_name = string(optarg);
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
    if (network_name.empty()) {
        network_name = file_name;
    }
    return make_tuple(file_name, path, network_name);
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
void showNodesInfo(bayesnet::Network& network, string className)
{
    cout << "Nodes:" << endl;
    for (auto [name, item] : network.getNodes()) {
        cout << "*" << item->getName() << " States -> " << item->getNumStates() << endl;
        cout << "-Parents:";
        for (auto parent : item->getParents()) {
            cout << " " << parent->getName();
        }
        cout << endl;
        cout << "-Children:";
        for (auto child : item->getChildren()) {
            cout << " " << child->getName();
        }
        cout << endl;
    }
}
void showCPDS(bayesnet::Network& network)
{
    cout << "CPDs:" << endl;
    auto nodes = network.getNodes();
    for (auto it = nodes.begin(); it != nodes.end(); it++) {
        cout << "* Name: " << it->first << " " << it->second->getName() << " -> " << it->second->getNumStates() << endl;
        cout << "Parents: ";
        for (auto parent : it->second->getParents()) {
            cout << parent->getName() << " -> " << parent->getNumStates() << ", ";
        }
        cout << endl;
        auto cpd = it->second->getCPT();
        cout << cpd << endl;
    }
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

pair<string, string> get_options(int argc, char** argv)
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
    string file_name;
    string path;
    string network_name;
    tie(file_name, path, network_name) = parse_arguments(argc, argv);
    if (datasets.find(file_name) == datasets.end()) {
        cout << "Invalid file name: " << file_name << endl;
        usage(argv[0]);
        exit(1);
    }
    file_name = path + file_name + ".arff";
    if (!file_exists(file_name)) {
        cout << "Data File " << file_name << " does not exist" << endl;
        usage(argv[0]);
        exit(1);
    }
    network_name = path + network_name + ".net";
    if (!file_exists(network_name)) {
        cout << "Network File " << network_name << " does not exist" << endl;
        usage(argv[0]);
        exit(1);
    }
    return { file_name, network_name };
}

void build_network(bayesnet::Network& network, string network_name, map<string, int> maxes)
{
    ifstream file(network_name);
    string line;
    while (getline(file, line)) {
        if (line[0] == '#') {
            continue;
        }
        istringstream iss(line);
        string parent, child;
        if (!(iss >> parent >> child)) {
            break;
        }
        network.addNode(parent, maxes[parent]);
        network.addNode(child, maxes[child]);
        network.addEdge(parent, child);
    }
    file.close();
}


int main(int argc, char** argv)
{
    string file_name, network_name;
    tie(file_name, network_name) = get_options(argc, argv);

    auto handler = ArffFiles();
    handler.load(file_name);
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
    cout << "Features: ";
    for (auto feature : features) {
        cout << "[" << feature << "] ";
    }
    cout << endl;
    cout << "Class name: " << className << endl;
    // Build Network
    auto network = bayesnet::Network(1.0);
    build_network(network, network_name, maxes);
    network.fit(Xd, y, features, className);
    cout << "Hello, Bayesian Networks!" << endl;
    showNodesInfo(network, className);
    //showCPDS(network);
    cout << "Score: " << network.score(Xd, y) << endl;
    cout << "PyTorch version: " << TORCH_VERSION << endl;
    cout << "BayesNet version: " << network.version() << endl;
    unsigned int nthreads = std::thread::hardware_concurrency();
    cout << "Computer has " << nthreads << " cores." << endl;
    cout << "conditionalEdgeWeight " << endl << network.conditionalEdgeWeight() << endl;
    return 0;
}