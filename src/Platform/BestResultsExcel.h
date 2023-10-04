#ifndef BESTRESULTS_EXCEL_H
#define BESTRESULTS_EXCEL_H
#include "ExcelFile.h"
#include <vector>
#include <map>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

namespace platform {

    class BestResultsExcel : ExcelFile {
    public:
        BestResultsExcel(const string& score, const vector<string>& models, const vector<string>& datasets, const json& table, const map<string, map<string, float>>& ranks, bool friedman, double significance);
        ~BestResultsExcel();
        void build();
        string getFileName();
    private:
        void header(bool ranks);
        void body(bool ranks);
        void footer(bool ranks);
        void formatColumns();
        void doFriedman();
        const string fileName = "BestResults.xlsx";
        string score;
        vector<string> models;
        vector<string> datasets;
        json table;
        map<string, map<string, float>> ranksModels;
        bool friedman;
        double significance;
        int modelNameSize = 12; // Min size of the column
        int datasetNameSize = 25; // Min size of the column
    };
}
#endif //BESTRESULTS_EXCEL_H