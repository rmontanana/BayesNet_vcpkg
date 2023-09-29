#ifndef DATASETS_H
#define DATASETS_H
#include <torch/torch.h>
#include <map>
#include <vector>
#include <string>
namespace platform {
    using namespace std;
    enum fileType_t { CSV, ARFF, RDATA };
    class SourceData {
    public:
        SourceData(string source)
        {
            if (source == "Surcov") {
                path = "datasets/";
                fileType = CSV;
            } else if (source == "Arff") {
                path = "datasets/";
                fileType = ARFF;
            } else if (source == "Tanveer") {
                path = "data/";
                fileType = RDATA;
            } else {
                throw invalid_argument("Unknown source.");
            }
        }
        string getPath()
        {
            return path;
        }
        fileType_t getFileType()
        {
            return fileType;
        }
    private:
        string path;
        fileType_t fileType;
    };
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
        void load_rdata();
        void computeStates();
    public:
        Dataset(const string& path, const string& name, const string& className, bool discretize, fileType_t fileType) : path(path), name(name), className(className), discretize(discretize), loaded(false), fileType(fileType) {};
        explicit Dataset(const Dataset&);
        string getName() const;
        string getClassName() const;
        vector<string> getFeatures() const;
        map<string, vector<int>> getStates() const;
        pair<vector<vector<float>>&, vector<int>&> getVectors();
        pair<vector<vector<int>>&, vector<int>&> getVectorsDiscretized();
        pair<torch::Tensor&, torch::Tensor&> getTensors();
        int getNFeatures() const;
        int getNSamples() const;
        void load();
        const bool inline isLoaded() const { return loaded; };
    };
    class Datasets {
    private:
        string path;
        fileType_t fileType;
        string sfileType;
        map<string, unique_ptr<Dataset>> datasets;
        bool discretize;
        void load(); // Loads the list of datasets
    public:
        explicit Datasets(bool discretize, string sfileType) : discretize(discretize), sfileType(sfileType) { load(); };
        vector<string> getNames();
        vector<string> getFeatures(const string& name) const;
        int getNSamples(const string& name) const;
        string getClassName(const string& name) const;
        int getNClasses(const string& name);
        vector<int> getClassesCounts(const string& name) const;
        map<string, vector<int>> getStates(const string& name) const;
        pair<vector<vector<float>>&, vector<int>&> getVectors(const string& name);
        pair<vector<vector<int>>&, vector<int>&> getVectorsDiscretized(const string& name);
        pair<torch::Tensor&, torch::Tensor&> getTensors(const string& name);
        bool isDataset(const string& name) const;
        void loadDataset(const string& name) const;
    };
};

#endif