#include <filesystem>
#include <fstream>
#include <sstream>
#include "Result.h"
#include "Colors.h"
#include "BestScore.h"
namespace platform {
    Result::Result(const string& path, const string& filename)
        : path(path)
        , filename(filename)
    {
        auto data = load();
        date = data["date"];
        score = 0;
        for (const auto& result : data["results"]) {
            score += result["score"].get<double>();
        }
        scoreName = data["score_name"];
        if (scoreName == BestScore::scoreName()) {
            score /= BestScore::score();
        }
        title = data["title"];
        duration = data["duration"];
        model = data["model"];
        complete = data["results"].size() > 1;
    }

    json Result::load() const
    {
        ifstream resultData(path + "/" + filename);
        if (resultData.is_open()) {
            json data = json::parse(resultData);
            return data;
        }
        throw invalid_argument("Unable to open result file. [" + path + "/" + filename + "]");
    }

    string Result::to_string() const
    {
        stringstream oss;
        oss << date << " ";
        oss << setw(12) << left << model << " ";
        oss << setw(11) << left << scoreName << " ";
        oss << right << setw(11) << setprecision(7) << fixed << score << " ";
        auto completeString = isComplete() ? "C" : "P";
        oss << setw(1) << " " << completeString << "  ";
        oss << setw(9) << setprecision(3) << fixed << duration << " ";
        oss << setw(50) << left << title << " ";
        return  oss.str();
    }
}