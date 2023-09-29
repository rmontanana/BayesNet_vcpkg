#ifndef DATASETS_H
#define DATASETS_H
#include "Dataset.h"
namespace platform {
    using namespace std;
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