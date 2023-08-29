#include "Datasets.h"
#include "platformUtils.h"
#include "ArffFiles.h"
#include <fstream>
namespace platform {
    void Datasets::load()
    {
        ifstream catalog(path + "/all.txt");
        if (catalog.is_open()) {
            string line;
            while (getline(catalog, line)) {
                vector<string> tokens = split(line, ',');
                string name = tokens[0];
                string className = tokens[1];
                datasets[name] = make_unique<Dataset>(path, name, className, discretize, fileType);
            }
            catalog.close();
        } else {
            throw invalid_argument("Unable to open catalog file. [" + path + "/all.txt" + "]");
        }
    }
    vector<string> Datasets::getNames()
    {
        vector<string> result;
        transform(datasets.begin(), datasets.end(), back_inserter(result), [](const auto& d) { return d.first; });
        return result;
    }
    vector<string> Datasets::getFeatures(const string& name) const
    {
        if (datasets.at(name)->isLoaded()) {
            return datasets.at(name)->getFeatures();
        } else {
            throw invalid_argument("Dataset not loaded.");
        }
    }
    map<string, vector<int>> Datasets::getStates(const string& name) const
    {
        if (datasets.at(name)->isLoaded()) {
            return datasets.at(name)->getStates();
        } else {
            throw invalid_argument("Dataset not loaded.");
        }
    }
    void Datasets::loadDataset(const string& name) const
    {
        if (datasets.at(name)->isLoaded()) {
            return;
        } else {
            datasets.at(name)->load();
        }
    }
    string Datasets::getClassName(const string& name) const
    {
        if (datasets.at(name)->isLoaded()) {
            return datasets.at(name)->getClassName();
        } else {
            throw invalid_argument("Dataset not loaded.");
        }
    }
    int Datasets::getNSamples(const string& name) const
    {
        if (datasets.at(name)->isLoaded()) {
            return datasets.at(name)->getNSamples();
        } else {
            throw invalid_argument("Dataset not loaded.");
        }
    }
    int Datasets::getNClasses(const string& name)
    {
        if (datasets.at(name)->isLoaded()) {
            auto className = datasets.at(name)->getClassName();
            if (discretize) {
                auto states = getStates(name);
                return states.at(className).size();
            }
            auto [Xv, yv] = getVectors(name);
            return *max_element(yv.begin(), yv.end()) + 1;
        } else {
            throw invalid_argument("Dataset not loaded.");
        }
    }
    vector<int> Datasets::getClassesCounts(const string& name) const
    {
        if (datasets.at(name)->isLoaded()) {
            auto [Xv, yv] = datasets.at(name)->getVectors();
            vector<int> counts(*max_element(yv.begin(), yv.end()) + 1);
            for (auto y : yv) {
                counts[y]++;
            }
            return counts;
        } else {
            throw invalid_argument("Dataset not loaded.");
        }
    }
    pair<vector<vector<float>>&, vector<int>&> Datasets::getVectors(const string& name)
    {
        if (!datasets[name]->isLoaded()) {
            datasets[name]->load();
        }
        return datasets[name]->getVectors();
    }
    pair<vector<vector<int>>&, vector<int>&> Datasets::getVectorsDiscretized(const string& name)
    {
        if (!datasets[name]->isLoaded()) {
            datasets[name]->load();
        }
        return datasets[name]->getVectorsDiscretized();
    }
    pair<torch::Tensor&, torch::Tensor&> Datasets::getTensors(const string& name)
    {
        if (!datasets[name]->isLoaded()) {
            datasets[name]->load();
        }
        return datasets[name]->getTensors();
    }
    bool Datasets::isDataset(const string& name) const
    {
        return datasets.find(name) != datasets.end();
    }
    Dataset::Dataset(const Dataset& dataset) : path(dataset.path), name(dataset.name), className(dataset.className), n_samples(dataset.n_samples), n_features(dataset.n_features), features(dataset.features), states(dataset.states), loaded(dataset.loaded), discretize(dataset.discretize), X(dataset.X), y(dataset.y), Xv(dataset.Xv), Xd(dataset.Xd), yv(dataset.yv), fileType(dataset.fileType)
    {
    }
    string Dataset::getName() const
    {
        return name;
    }
    string Dataset::getClassName() const
    {
        return className;
    }
    vector<string> Dataset::getFeatures() const
    {
        if (loaded) {
            return features;
        } else {
            throw invalid_argument("Dataset not loaded.");
        }
    }
    int Dataset::getNFeatures() const
    {
        if (loaded) {
            return n_features;
        } else {
            throw invalid_argument("Dataset not loaded.");
        }
    }
    int Dataset::getNSamples() const
    {
        if (loaded) {
            return n_samples;
        } else {
            throw invalid_argument("Dataset not loaded.");
        }
    }
    map<string, vector<int>> Dataset::getStates() const
    {
        if (loaded) {
            return states;
        } else {
            throw invalid_argument("Dataset not loaded.");
        }
    }
    pair<vector<vector<float>>&, vector<int>&> Dataset::getVectors()
    {
        if (loaded) {
            return { Xv, yv };
        } else {
            throw invalid_argument("Dataset not loaded.");
        }
    }
    pair<vector<vector<int>>&, vector<int>&> Dataset::getVectorsDiscretized()
    {
        if (loaded) {
            return { Xd, yv };
        } else {
            throw invalid_argument("Dataset not loaded.");
        }
    }
    pair<torch::Tensor&, torch::Tensor&> Dataset::getTensors()
    {
        if (loaded) {
            buildTensors();
            return { X, y };
        } else {
            throw invalid_argument("Dataset not loaded.");
        }
    }
    void Dataset::load_csv()
    {
        ifstream file(path + "/" + name + ".csv");
        if (file.is_open()) {
            string line;
            getline(file, line);
            vector<string> tokens = split(line, ',');
            features = vector<string>(tokens.begin(), tokens.end() - 1);
            className = tokens.back();
            for (auto i = 0; i < features.size(); ++i) {
                Xv.push_back(vector<float>());
            }
            while (getline(file, line)) {
                tokens = split(line, ',');
                for (auto i = 0; i < features.size(); ++i) {
                    Xv[i].push_back(stof(tokens[i]));
                }
                yv.push_back(stoi(tokens.back()));
            }
            file.close();
        } else {
            throw invalid_argument("Unable to open dataset file.");
        }
    }
    void Dataset::computeStates()
    {
        for (int i = 0; i < features.size(); ++i) {
            states[features[i]] = vector<int>(*max_element(Xd[i].begin(), Xd[i].end()) + 1);
            iota(begin(states[features[i]]), end(states[features[i]]), 0);
        }
        states[className] = vector<int>(*max_element(yv.begin(), yv.end()) + 1);
        iota(begin(states[className]), end(states[className]), 0);
    }
    void Dataset::load_arff()
    {
        auto arff = ArffFiles();
        arff.load(path + "/" + name + ".arff", className);
        // Get Dataset X, y
        Xv = arff.getX();
        yv = arff.getY();
        // Get className & Features
        className = arff.getClassName();
        auto attributes = arff.getAttributes();
        transform(attributes.begin(), attributes.end(), back_inserter(features), [](const auto& attribute) { return attribute.first; });
    }
    void Dataset::load()
    {
        if (loaded) {
            return;
        }
        if (fileType == CSV) {
            load_csv();
        } else if (fileType == ARFF) {
            load_arff();
        }
        if (discretize) {
            Xd = discretizeDataset(Xv, yv);
            computeStates();
        }
        n_samples = Xv[0].size();
        n_features = Xv.size();
        loaded = true;
    }
    void Dataset::buildTensors()
    {
        if (discretize) {
            X = torch::zeros({ static_cast<int>(n_features), static_cast<int>(n_samples) }, torch::kInt32);
        } else {
            X = torch::zeros({ static_cast<int>(n_features), static_cast<int>(n_samples) }, torch::kFloat32);
        }
        for (int i = 0; i < features.size(); ++i) {
            if (discretize) {
                X.index_put_({ i,  "..." }, torch::tensor(Xd[i], torch::kInt32));
            } else {
                X.index_put_({ i,  "..." }, torch::tensor(Xv[i], torch::kFloat32));
            }
        }
        y = torch::tensor(yv, torch::kInt32);
    }
}