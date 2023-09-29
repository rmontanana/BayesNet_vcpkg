#include "Datasets.h"
#include <fstream>
namespace platform {
    void Datasets::load()
    {
        auto sd = SourceData(sfileType);
        fileType = sd.getFileType();
        path = sd.getPath();
        ifstream catalog(path + "all.txt");
        if (catalog.is_open()) {
            string line;
            while (getline(catalog, line)) {
                if (line.empty() || line[0] == '#') {
                    continue;
                }
                vector<string> tokens = Dataset::split(line, ',');
                string name = tokens[0];
                string className;
                try {
                    className = tokens[1];
                }
                catch (exception e) {
                    className = "-1";
                }
                datasets[name] = make_unique<Dataset>(path, name, className, discretize, fileType);
            }
            catalog.close();
        } else {
            throw invalid_argument("Unable to open catalog file. [" + path + "all.txt" + "]");
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
}