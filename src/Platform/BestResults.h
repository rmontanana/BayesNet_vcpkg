#ifndef BESTRESULTS_H
#define BESTRESULTS_H
#include <string>
using namespace std;

namespace platform {
    class BestResults {
    public:
        explicit BestResults(const string& path, const string& score, const string& model) : path(path), score(score), model(model) {}
        void build();
        void report();
    private:
        vector<string> loadFiles();
        string bestResultFile();
        string path;
        string score;
        string model;
    };
}
#endif //BESTRESULTS_H