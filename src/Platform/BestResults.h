#ifndef BESTRESULTS_H
#define BESTRESULTS_H
#include <string>
#include <nlohmann/json.hpp>
using namespace std;
using json = nlohmann::json;
namespace platform {
    class BestResults {
    public:
        explicit BestResults(const string& path, const string& score, const string& model) : path(path), score(score), model(model) {}
        string build();
        void report();
    private:
        vector<string> loadFiles();
        string bestResultFile();
        json loadFile(const string& fileName);
        string path;
        string score;
        string model;
    };
}
#endif //BESTRESULTS_H