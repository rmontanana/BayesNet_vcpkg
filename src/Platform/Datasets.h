#ifndef DATASETS_H
#define DATASETS_H
#include <torch/torch.h>
#include <map>
#include <vector>
#include <string>
namespace platform {
    using namespace std;
    enum fileType_t { CSV, ARFF };
    class Dataset {
    private:
        string path;
        string name;
        fileType_t fileType;
        string className;
        int n_samples{ 0 }, n_features{ 0 };
        vector<string> features;
        map<string, vector<int>> states;
        bool loaded;
        bool discretize;
        torch::Tensor X, y;
        vector<vector<float>> Xv;
        vector<vector<int>> Xd;
        vector<int> yv;
        void buildTensors();
        void load_csv();
        void load_arff();
        void computeStates();
    public:
        Dataset(const string& path, const string& name, const string& className, bool discretize, fileType_t fileType) : path(path), name(name), className(className), discretize(discretize), loaded(false), fileType(fileType) {};
        explicit Dataset(const Dataset&);
        string getName();
        string getClassName();
        vector<string> getFeatures();
        map<string, vector<int>> getStates();
        pair<vector<vector<float>>&, vector<int>&> getVectors();
        pair<vector<vector<int>>&, vector<int>&> getVectorsDiscretized();
        pair<torch::Tensor&, torch::Tensor&> getTensors();
        int getNFeatures();
        int getNSamples();
        void load();
        const bool inline isLoaded() const { return loaded; };
    };
    class Datasets {
    private:
        string path;
        fileType_t fileType;
        map<string, unique_ptr<Dataset>> datasets;
        bool discretize;
        void load(); // Loads the list of datasets
    public:
        Datasets(const string& path, bool discretize = false, fileType_t fileType = ARFF) : path(path), discretize(discretize), fileType(fileType) { load(); };
        vector<string> getNames();
        vector<string> getFeatures(string name);
        int getNSamples(string name);
        string getClassName(string name);
        map<string, vector<int>> getStates(string name);
        pair<vector<vector<float>>&, vector<int>&> getVectors(string name);
        pair<vector<vector<int>>&, vector<int>&> getVectorsDiscretized(string name);
        pair<torch::Tensor&, torch::Tensor&> getTensors(string name);
        bool isDataset(string name);
    };
};

#endif