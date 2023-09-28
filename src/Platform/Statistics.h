#ifndef STATISTICS_H
#define STATISTICS_H
#include <iostream>
#include <vector>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

namespace platform {
    struct WTL {
        int win;
        int tie;
        int loss;
    };
    class Statistics {
    public:
        Statistics(vector<string>& models, vector<string>& datasets, json data, double significance = 0.05);
        bool friedmanTest();
        void postHocHolmTest(bool friedmanResult);
    private:
        void fit();
        void computeRanks();
        void computeWTL();
        vector<string> models;
        vector<string> datasets;
        json data;
        double significance;
        bool fitted = false;
        int nModels = 0;
        int nDatasets = 0;
        int controlIdx = 0;
        map<int, WTL> wtl;
        map<string, float> ranks;
        int maxModelName = 0;
        int maxDatasetName = 0;
    };
}
#endif // !STATISTICS_H