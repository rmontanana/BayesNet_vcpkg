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
        json load() const;
        string to_string() const;
        string getFilename() const { return filename; };
        string getDate() const { return date; };
        double getScore() const { return score; };
        string getTitle() const { return title; };
        double getDuration() const { return duration; };
        string getModel() const { return model; };
        string getScoreName() const { return scoreName; };
        bool isComplete() const { return complete; };
    private:
        string path;
        string filename;
        string date;
        double score;
        string title;
        double duration;
        string model;
        string scoreName;
        bool complete;
    };
    class Results {
    public:
        Results(const string& path, const int max, const string& model, const string& score, bool complete, bool partial) : path(path), max(max), model(model), scoreName(score), complete(complete), partial(partial) { load(); };
        void manage();
    private:
        string path;
        int max;
        string model;
        string scoreName;
        bool complete;
        bool partial;
        bool indexList = true;
        vector<Result> files;
        void load(); // Loads the list of results
        void show() const;
        void report(const int index, const bool excelReport) const;
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