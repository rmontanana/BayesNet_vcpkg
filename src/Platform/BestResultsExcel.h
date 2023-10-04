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
        BestResultsExcel(const string& score, const vector<string>& models, const vector<string>& datasets, const json& table, bool friedman, double significance);
        ~BestResultsExcel();
        void build();
        string getFileName();
    private:
        void header();
        void body();
        void footer();
        void formatColumns();
        const string fileName = "BestResults.xlsx";
        const string& score;
        const vector<string>& models;
        const vector<string>& datasets;
        const json& table;
        bool friedman;
        double significance;
        int modelNameSize = 12; // Min size of the column
        int datasetNameSize = 25; // Min size of the column
        // map<string, map<string, float>>& ranksModels;
    };
}
#endif //BESTRESULTS_EXCEL_H