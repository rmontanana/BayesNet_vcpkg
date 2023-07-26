#include "Datasets.h"
#include "platformUtils.h"
#include "ArffFiles.h"
namespace platform {
    vector<string> split(string text, char delimiter)
    {
        vector<string> result;
        stringstream ss(text);
        string token;
        while (getline(ss, token, delimiter)) {
            result.push_back(token);
        }
        return result;
    }
    void Datasets::load()
    {
        string line;
        ifstream catalog(path + "/all.txt");
        if (catalog.is_open()) {
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
    Dataset& Datasets::getDataset(string name)
    {
        if (datasets.find(name) == datasets.end()) {
            throw invalid_argument("Dataset not found.");
        }
        return *datasets[name];
    }
    vector<string> Datasets::getNames()
    {
        vector<string> result;
        for (auto& d : datasets) {
            result.push_back(d.first);
        }
        return result;
    }
    vector<string> Datasets::getFeatures(string name)
    {
        auto dataset = getDataset(name);
        if (dataset.isLoaded()) {
            return dataset.getFeatures();
        } else {
            throw invalid_argument("Dataset not loaded.");
        }
    }
    map<string, vector<int>> Datasets::getStates(string name)
    {
        auto dataset = getDataset(name);
        if (dataset.isLoaded()) {
            return dataset.getStates();
        } else {
            throw invalid_argument("Dataset not loaded.");
        }
    }
    pair<vector<vector<float>>&, vector<int>&> Datasets::getVectors(string name)
    {
        auto dataset = getDataset(name);
        if (!dataset.isLoaded()) {
            dataset.load();
        }
        return dataset.getVectors();
    }
    pair<vector<vector<int>>&, vector<int>&> Datasets::getVectorsDiscretized(string name)
    {
        auto dataset = getDataset(name);
        if (!dataset.isLoaded()) {
            dataset.load();
        }
        return dataset.getVectorsDiscretized();
    }
    pair<torch::Tensor&, torch::Tensor&> Datasets::getTensors(string name)
    {
        auto dataset = getDataset(name);
        if (!dataset.isLoaded()) {
            dataset.load();
        }
        return dataset.getTensors();
    }
    Dataset::Dataset(Dataset& dataset)
    {
        path = dataset.path;
        name = dataset.name;
        className = dataset.className;
        n_samples = dataset.n_samples;
        n_features = dataset.n_features;
        features = dataset.features;
        states = dataset.states;
        loaded = dataset.loaded;
        discretize = dataset.discretize;
        X = dataset.X;
        y = dataset.y;
        Xv = dataset.Xv;
        Xd = dataset.Xd;
        yv = dataset.yv;
        fileType = dataset.fileType;
    }
    string Dataset::getName()
    {
        return name;
    }
    string Dataset::getClassName()
    {
        return className;
    }
    vector<string> Dataset::getFeatures()
    {
        if (loaded) {
            return features;
        } else {
            throw invalid_argument("Dataset not loaded.");
        }
    }
    int Dataset::getNFeatures()
    {
        if (loaded) {
            return n_features;
        } else {
            throw invalid_argument("Dataset not loaded.");
        }
    }
    int Dataset::getNSamples()
    {
        if (loaded) {
            return n_samples;
        } else {
            throw invalid_argument("Dataset not loaded.");
        }
    }
    map<string, vector<int>> Dataset::getStates()
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
        string line;
        ifstream file(path + "/" + name + ".csv");
        if (file.is_open()) {
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
            states[features[i]] = vector<int>(*max_element(Xd[i].begin(), Xd[i].end()));
            iota(Xd[i].begin(), Xd[i].end(), 0);
        }
        states[className] = vector<int>(*max_element(yv.begin(), yv.end()));
        iota(yv.begin(), yv.end(), 0);
    }
    void Dataset::load_arff()
    {
        auto arff = ArffFiles();
        arff.load(path + "/" + name + ".arff", className);
        // Get Dataset X, y
        Xv = arff.getX();
        yv = arff.getY();
        // Get className & Features
        auto className = arff.getClassName();
        vector<string> features;
        for (auto feature : arff.getAttributes()) {
            features.push_back(feature.first);
        }
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
            n_samples = Xd[0].size();
            n_features = Xd.size();
        }
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
            y = torch::tensor(yv, torch::kInt32);
        }
    }
}