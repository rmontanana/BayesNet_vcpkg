#include "Result.h"
#include "BestScore.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include "Colors.h"
#include "DotEnv.h"
#include "CLocale.h"

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
        auto best = BestScore::getScore(scoreName);
        if (best.first != "") {
            score /= best.second;
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

    string Result::to_string(int maxModel) const
    {
        auto tmp = ConfigLocale();
        stringstream oss;
        double durationShow = duration > 3600 ? duration / 3600 : duration > 60 ? duration / 60 : duration;
        string durationUnit = duration > 3600 ? "h" : duration > 60 ? "m" : "s";
        oss << date << " ";
        oss << setw(maxModel) << left << model << " ";
        oss << setw(11) << left << scoreName << " ";
        oss << right << setw(11) << setprecision(7) << fixed << score << " ";
        auto completeString = isComplete() ? "C" : "P";
        oss << setw(1) << " " << completeString << "  ";
        oss << setw(7) << setprecision(2) << fixed << durationShow << " " << durationUnit << " ";
        oss << setw(50) << left << title << " ";
        return  oss.str();
    }
}