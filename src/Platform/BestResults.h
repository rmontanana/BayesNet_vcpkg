#ifndef BESTRESULTS_H
#define BESTRESULTS_H
#include <string>
#include <nlohmann/json.hpp>
using namespace std;
using json = nlohmann::json;
namespace platform {
    class BestResults {
    public:
        explicit BestResults(const string& path, const string& score, const string& model, bool friedman, double significance = 0.05) : path(path), score(score), model(model), friedman(friedman), significance(significance) {}
        string build();
        void reportSingle();
        void reportAll(bool excel);
        void buildAll();
    private:
        vector<string> getModels();
        vector<string> getDatasets(json table);
        vector<string> loadResultFiles();
        json buildTableResults(vector<string> models);
        void printTableResults(vector<string> models, json table);
        string bestResultFile();
        json loadFile(const string& fileName);
        string path;
        string score;
        string model;
        bool friedman;
        double significance;
        int maxModelName = 0;
        int maxDatasetName = 0;
    };
}
#endif //BESTRESULTS_H