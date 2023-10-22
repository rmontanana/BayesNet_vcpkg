#ifndef RESULT_H
#define RESULT_H
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
        string to_string(int maxModel) const;
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
};

#endif