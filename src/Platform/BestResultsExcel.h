#ifndef BESTRESULTS_EXCEL_H
#define BESTRESULTS_EXCEL_H
#include <vector>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

namespace platform {
    class BestResultsExcel {
    public:
        BestResultsExcel(vector<string> models, vector<string> datasets, json table, bool friedman) : models(models), datasets(datasets), table(table), friedman(friedman) {}
        void build();
    private:
        vector<string> models;
        vector<string> datasets;
        json table;
        bool friedman;
    };
}
#endif //BESTRESULTS_EXCEL_H