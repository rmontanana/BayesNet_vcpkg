#ifndef RESULTS_H
#define RESULTS_H
#include <map>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include "Result.h"
namespace platform {
    using namespace std;
    using json = nlohmann::json;

    class Results {
    public:
        Results(const string& path, const string& model, const string& score, bool complete, bool partial);
        void sortDate();
        void sortScore();
        void sortModel();
        void sortDuration();
        int maxModelSize() const { return maxModel; };
        void hideResult(int index, const string& pathHidden);
        void deleteResult(int index);
        int size() const;
        bool empty() const;
        vector<Result>::iterator begin() { return files.begin(); };
        vector<Result>::iterator end() { return files.end(); };
        Result& at(int index) { return files.at(index); };
    private:
        string path;
        string model;
        string scoreName;
        bool complete;
        bool partial;
        int maxModel;
        vector<Result> files;
        void load(); // Loads the list of results
    };
};

#endif