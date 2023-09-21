#ifndef RESULTS_H
#define RESULTS_H
#include "xlsxwriter.h"
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
        Results(const string& path, const int max, const string& model, const string& score, bool complete, bool partial, bool compare) :
            path(path), max(max), model(model), scoreName(score), complete(complete), partial(partial), compare(compare)
        {
            load();
        };
        void manage();
    private:
        string path;
        int max;
        string model;
        string scoreName;
        bool complete;
        bool partial;
        bool indexList = true;
        bool openExcel = false;
        bool compare;
        lxw_workbook* workbook = NULL;
        vector<Result> files;
        void load(); // Loads the list of results
        void show() const;
        void report(const int index, const bool excelReport);
        void showIndex(const int index, const int idx) const;
        int getIndex(const string& intent) const;
        void menu();
        void sortList();
        void sortDate();
        void sortScore();
        void sortModel();
        void sortDuration();
    };
};

#endif