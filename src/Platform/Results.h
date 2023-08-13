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
        string getFilename() const { return filename; };
        string getDate() const { return date; };
        double getScore() const { return score; };
        string getTitle() const { return title; };
        double getDuration() const { return duration; };
        string getModel() const { return model; };
        string getScoreName() const { return scoreName; };
    private:
        string path;
        string filename;
        string date;
        double score;
        string title;
        double duration;
        string model;
        string scoreName;
    };
    class Results {
    public:
        Results(const string& path, const int max, const string& model, const string& score) : path(path), max(max), model(model), scoreName(score) { load(); };
        void manage();
    private:
        string path;
        int max;
        string model;
        string scoreName;
        vector<Result> files;
        void load(); // Loads the list of results
        void show() const;
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