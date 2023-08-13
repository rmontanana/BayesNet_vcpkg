#ifndef RESULTS_H
#define RESULTS_H
#include <map>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
namespace platform {
    using namespace std;
    using json = nlohmann::json;

    class Result {
    public:
        Result(const string& path, const string& filename);
        json load();
        string to_string() const;
    private:
        string path;
        string filename;
        string date;
        double score;
        string title;
        double duration;
        string model;
    };
    class Results {
    public:
        explicit Results(const string& path) : path(path) { load(); };
        void manage();
    private:
        string path;
        vector<Result> files;
        void load(); // Loads the list of results
        void show();
        int menu();
    };
};

#endif